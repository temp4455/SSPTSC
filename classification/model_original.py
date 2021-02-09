import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import numpy as np

def clones(moudel, N):
    return nn.ModuleList([copy.deepcopy(moudel) for _ in range(N)])


# class Encoder(nn.Module):
#     def __init__(self,layer,N):
#         super(Encoder, self).__init__()
#         self.layers=clones(layer,N)
#         self.norm=LayerNorm(layer.size)
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x=layer(x,mask)
#         return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.view(batch_size, channels * seq_len)
        # x1: [batch, channel, seq_len]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.a_2 * (x - mean) / (std + self.eps) + self.b_2).view(batch_size, channels, seq_len)


# class SublayerConnection(nn.Module):
#     def __init__(self,features,dropout):
#         super(SublayerConnection,self).__init__()
#         # self.norm=LayerNorm(size)
#         self.norm= nn.BatchNorm1d(features)
#         self.dropout=nn.Dropout(dropout)
#     def forward(self, x,sublayer):
#         return x+self.dropout(sublayer(self.norm(x)))


# class EncoderLayer(nn.Module):
#     def __init__(self,size,self_attn,feed_forward,dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn=self_attn
#         self.feed_forward= feed_forward
#         self.sublayer= clones(SublayerConnection(size,dropout),2)
#         self.size= size
#     def forward(self, x, mask):
#         x= self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
#         return  self.sublayer[1](x,self.feed_forward)
def attention(query: torch.Tensor, key: torch.Tensor, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    # p_attn attention
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def _chunk(hidden_states, window_overlap):
    """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8


def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
    """pads rows and then flips rows and columns"""
    hidden_states_padded = F.pad(
        hidden_states_padded, padding
    )  # padding value is not important because it will be overwritten
    hidden_states_padded = hidden_states_padded.view(
        *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
    )
    return hidden_states_padded


def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8


def _sliding_chunks_query_key_matmul( query: torch.Tensor, key: torch.Tensor, window_overlap: int):
    """Matrix multiplication of query and key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size window_overlap"""
    batch_size, seq_len, num_heads, head_dim = query.size()
    assert (
            seq_len % (window_overlap * 2) == 0
    ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
    assert query.size() == key.size()

    chunks_count = seq_len // window_overlap - 1

    # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
    query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

    chunked_query = _chunk(query, window_overlap)
    chunked_key = _chunk(key, window_overlap)

    # matrix multipication
    # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
    # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
    chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunked_query, chunked_key))  # multiply

    # convert diagonals into columns
    diagonal_chunked_attention_scores = _pad_and_transpose_last_two_dims(
        chunked_attention_scores, padding=(0, 0, 0, 1)
    )

    # allocate space for the overall attention matrix where the chunks are combined. The last dimension
    # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
    # window_overlap previous words). The following column is attention score from each word to itself, then
    # followed by window_overlap columns for the upper triangle.

    diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
        (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
    )

    # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                            :, :, :window_overlap, : window_overlap + 1
                                                            ]
    diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                           :, -1, window_overlap:, : window_overlap + 1
                                                           ]
    # - copying the lower triangle
    diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                           :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                           ]

    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                          :, 0, : window_overlap - 1,
                                                                          1 - window_overlap:
                                                                          ]

    # separate batch_size and num_heads dimensions again
    diagonal_attention_scores = diagonal_attention_scores.view(
        batch_size, num_heads, seq_len, 2 * window_overlap + 1
    ).transpose(2, 1)

    _mask_invalid_locations(diagonal_attention_scores, window_overlap)
    return diagonal_attention_scores


def _pad_and_diagonalize(chunked_hidden_states):
    """shift every row 1 step right, converting columns into diagonals.
    Example:
          chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                   -1.8348,  0.7672,  0.2986,  0.0285,
                                   -0.7584,  0.4206, -0.0405,  0.1599,
                                   2.0514, -1.1600,  0.5372,  0.2629 ]
          window_overlap = num_rows = 4
         (pad & diagonilize) =>
         [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
           0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
           0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
           0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
    """
    total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
    chunked_hidden_states = F.pad(
        chunked_hidden_states, (0, window_overlap + 1)
    )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, -1
    )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
    chunked_hidden_states = chunked_hidden_states[
                            :, :, :-window_overlap
                            ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
    )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    return chunked_hidden_states


def _sliding_chunks_matmul_attn_probs_value(
        attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
):
    """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
    Returned tensor will be of the same shape as `attn_probs`"""
    batch_size, seq_len, num_heads, head_dim = value.size()

    assert seq_len % (window_overlap * 2) == 0
    assert attn_probs.size()[:3] == value.size()[:3]
    assert attn_probs.size(3) == 2 * window_overlap + 1
    chunks_count = seq_len // window_overlap - 1
    # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

    chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
    )

    # group batch_size and num_heads dimensions into one
    value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

    # pad seq_len with w at the beginning of the sequence and another window overlap at the end
    padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

    # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
    chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
    chunked_value_stride = padded_value.stride()
    chunked_value_stride = (
        chunked_value_stride[0],
        window_overlap * chunked_value_stride[1],
        chunked_value_stride[1],
        chunked_value_stride[2],
    )
    chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

    chunked_attn_probs = _pad_and_diagonalize(chunked_attn_probs)

    context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)


def _get_global_attn_indices(is_index_global_attn):
    """ compute global attn indices required throughout forward pass """
    # helper variable
    num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

    # max number of global attn indices in batch
    max_num_global_attn_indices = num_global_attn_indices.max()

    # indices of global attn
    # print(is_index_global_attn)
    is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

    # helper variable
    is_local_index_global_attn = torch.arange(
        max_num_global_attn_indices, device=is_index_global_attn.device
    ) < num_global_attn_indices.unsqueeze(dim=-1)

    # location of the non-padding values within global attention indices
    is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

    # location of the padding values within global attention indices
    is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
    return (
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    )


def _concat_with_global_key_attn_probs(
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        num_heads,
        head_dim,
):
    batch_size = key_vectors.shape[0]

    # create only global key vectors
    key_vectors_only_global = key_vectors.new_zeros(
        batch_size, max_num_global_attn_indices, num_heads, head_dim
    )

    key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

    # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
    attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

    attn_probs_from_global_key[
    is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
    ] = -10000.0

    return attn_probs_from_global_key


def _compute_attn_output_with_global_indices(
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        num_heads,
        head_dim,
        one_sided_attn_window_size
):
    batch_size = attn_probs.shape[0]

    # cut local attn probs to global only
    attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
    # get value vectors for global only
    value_vectors_only_global = value_vectors.new_zeros(
        batch_size, max_num_global_attn_indices, num_heads, head_dim
    )
    value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

    # use `matmul` because `einsum` crashes sometimes with fp16
    # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
    # compute attn output only global
    attn_output_only_global = torch.matmul(
        attn_probs_only_global.transpose(1, 2), value_vectors_only_global.transpose(1, 2)
    ).transpose(1, 2)

    # reshape attn probs
    attn_probs_without_global = attn_probs.narrow(
        -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
    ).contiguous()

    # compute attn output with global
    attn_output_without_global = _sliding_chunks_matmul_attn_probs_value(
        attn_probs_without_global, value_vectors, one_sided_attn_window_size
    )
    return attn_output_only_global + attn_output_without_global


def _compute_global_attn_output_from_hidden(
        # hidden_states,
        query,
        key,
        value,
        max_num_global_attn_indices,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
        head_dim,
        num_heads,
        dropout,
        training,
):
    # [ seq_len,batch, embed_dim]
    # hidden_states:[seq_len , batch_size , embed_dim]
    seq_len, batch_size = key.shape[:2]

    # prepare global hidden states

    global_query_vectors_only_global = query
    global_key_vectors = key
    global_value_vectors = value
    # global key, query, value
    # global_query_vectors_only_global, global_key_vectors,global_value_vectors: [seq_len , batch_size , embed_dim]
    # global_query_vectors_only_global = query_global(global_attn_hidden_states)
    # global_key_vectors = key_global(hidden_states)
    # global_value_vectors = value_global(hidden_states)

    # normalize
    global_query_vectors_only_global /= math.sqrt(head_dim)

    # reshape
    global_query_vectors_only_global = (
        global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * num_heads, head_dim)
            .transpose(0, 1)
    )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
    global_key_vectors = (
        global_key_vectors.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    )  # batch_size * self.num_heads, seq_len, head_dim)
    global_value_vectors = (
        global_value_vectors.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    )  # batch_size * self.num_heads, seq_len, head_dim)

    # compute attn scores
    global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))

    assert list(global_attn_scores.size()) == [
        batch_size * num_heads,
        max_num_global_attn_indices,
        seq_len,
    ], f"global_attn_scores have the wrong size. Size should be {(batch_size * num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}."

    global_attn_scores = global_attn_scores.view(batch_size, num_heads, max_num_global_attn_indices, seq_len)

    global_attn_scores[
    is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
    ] = -10000.0

    global_attn_scores = global_attn_scores.masked_fill(
        is_index_masked[:, None, None, :],
        -10000.0,
    )

    global_attn_scores = global_attn_scores.view(batch_size * num_heads, max_num_global_attn_indices, seq_len)

    # compute global attn probs
    global_attn_probs_float = F.softmax(
        global_attn_scores, dim=-1, dtype=torch.float32
    )  # use fp32 for numerical stability

    global_attn_probs = F.dropout(
        global_attn_probs_float.type_as(global_attn_scores), p=dropout, training=training
    )

    # global attn output
    global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

    assert list(global_attn_output.size()) == [
        batch_size * num_heads,
        max_num_global_attn_indices,
        head_dim,
    ], f"global_attn_output tensor has the wrong size. Size should be {(batch_size * num_heads, max_num_global_attn_indices, head_dim)}, but is {global_attn_output.size()}."

    global_attn_output = global_attn_output.view(
        batch_size, num_heads, max_num_global_attn_indices, head_dim
    )
    return global_attn_output


def attention_longformer(query: torch.Tensor,
                         key: torch.Tensor,
                         value,
                         query_global,
                         key_global,
                         value_global,
                         attention_mask,
                         attention_window,
                         training,
                         dropout=None, output_attentions=False):
    """
    :param training:
    :param query: [batch,h,seq_len,d_model//h]
    :param key:  [batch,h,seq_len,d_model//h]
    :param value: [batch,h,seq_len,d_model//h]
    :param attention_mask: [batch_size, seq_len],
            -ve: no attention
              0: local attention
            +ve: global attention
    :param attention_window:
    :param dropout:  int
    :return:
    """
    d_k = query.size(-1)
    # [batch, h, seq_len, d_model//h]
    query /= math.sqrt(d_k)
    one_sided_attn_window_size = attention_window // 2
    batch_size, num_heads, seq_len, head_dim = query.shape
    embed_dim = head_dim * num_heads
    is_index_masked = attention_mask < 0
    is_index_global_attn = attention_mask > 0
    is_global_attn = is_index_global_attn.flatten().any().item()
    # [batch, seq_len, h, d_model//h]
    query, key, value = query.permute(0, 2, 1, 3).contiguous(), key.permute(0, 2, 1, 3).contiguous(), value.permute(0,2,1,3).contiguous()
    attn_scores = _sliding_chunks_query_key_matmul(query, key, one_sided_attn_window_size)
    remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
    float_mask = remove_from_windowed_attention_mask.type_as(query).masked_fill(remove_from_windowed_attention_mask,-10000.0)
    diagonal_mask = _sliding_chunks_query_key_matmul(float_mask.new_ones(size=float_mask.size()), float_mask,
                                                     one_sided_attn_window_size)
    attn_scores += diagonal_mask
    if is_global_attn:
        (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero,
         is_local_index_no_global_attn_nonzero,) = _get_global_attn_indices(is_index_global_attn)
        global_key_attn_scores = _concat_with_global_key_attn_probs(
            query_vectors=query,
            key_vectors=key,
            max_num_global_attn_indices=max_num_global_attn_indices,
            is_index_global_attn_nonzero=is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            num_heads=num_heads,
            head_dim=head_dim
        )
        attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
        # free memory
        del global_key_attn_scores

    attn_probs_fp32 = F.softmax(attn_scores, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
    attn_probs = attn_probs_fp32.type_as(attn_scores)
    # free memory
    del attn_probs_fp32
    # softmax sometimes inserts NaN if all positions are masked, replace them with 0
    # attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    attn_probs = F.dropout(attn_probs, p=dropout, training=training)
    # query, key, value = query.view((batch_size,num_heads,))

    # compute local attention output with global attention value and add
    if is_global_attn:
        # compute sum of global and local attn
        attn_output = _compute_attn_output_with_global_indices(
            value_vectors=value,
            attn_probs=attn_probs,
            max_num_global_attn_indices=max_num_global_attn_indices,
            is_index_global_attn_nonzero=is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            num_heads=num_heads,
            head_dim=head_dim,
            one_sided_attn_window_size=one_sided_attn_window_size
        )
    else:
        # (batch_size,  seq_len, num_heads,head_dim)
        # compute local attn only
        attn_output = _sliding_chunks_matmul_attn_probs_value(
            attn_probs, value, one_sided_attn_window_size
        )

    # (seq_len ,batch_size, embed_dim)
    attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

    # compute value for global attention and overwrite to attention output
    # TODO: remove the redundant computation

    if is_global_attn:
        # query: [batch, seq_len, h, d_model//h] -> query_global_vector :[ seq_len ,batch, embed_dim]
        query_global_vector = query_global(query.transpose(0, 1).reshape(seq_len, batch_size, embed_dim))
        # query_global_vector= F.dropout(query_global_vector,p=dropout,training=training)
        global_attn_hidden_states = query_global_vector.new_zeros(max_num_global_attn_indices, batch_size, embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = query_global_vector[
            is_index_global_attn_nonzero[::-1]
        ]
        key_global_vector = key_global(key.transpose(0, 1).reshape(seq_len, batch_size, embed_dim))
        value_global_vector = value_global(value.transpose(0, 1).reshape(seq_len, batch_size, embed_dim))
        # key_global_vector=F.dropout(key_global_vector,p=dropout,training=training)
        # value_global_vector= F.dropout(value_global_vector,p=dropout,training=training)

        global_attn_output = _compute_global_attn_output_from_hidden(
            query=global_attn_hidden_states,
            key=key_global_vector,
            value=value_global_vector,
            max_num_global_attn_indices=max_num_global_attn_indices,
            is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            is_index_global_attn_nonzero=is_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            is_index_masked=is_index_masked,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            training=training,
        )

        # get only non zero global attn output
        nonzero_global_attn_output = global_attn_output[
                                     is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
                                     ]

        # overwrite values with global attention
        attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
            len(is_local_index_global_attn_nonzero[0]), -1
        )

    # want:[batch, h, seq_len, d_model // h], fact
    # (seq_len ,batch_size, embed_dim)->( batch_size,seq_len, embed_dim)->[batch, h, seq_len, d_model // h]
    attn_output = attn_output.transpose(0, 1).reshape(batch_size,seq_len,num_heads,head_dim).transpose(1,2)

    if output_attentions:
        if is_global_attn:
            # With global attention, return global attention probabilities only
            # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
            # which is the attention weights from tokens with global attention to all tokens
            # It doesn't not return local attention
            # In case of variable number of global attantion in the rows of a batch,
            # attn_probs are padded with -10000.0 attention scores
            attn_probs = attn_probs.view(batch_size, num_heads, max_num_global_attn_indices, seq_len)
        else:
            # without global attention, return local attention probabilities
            # batch_size x num_heads x sequence_length x window_size
            # which is the attention weights of every token attending to its neighbours
            attn_probs = attn_probs.permute(0, 2, 1, 3)

    outputs = (attn_output, attn_probs) if output_attentions else (attn_output,None)
    return outputs


class BasicLayer(nn.Module):
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]

    def __init__(self, in_channels, planes, stride,global_index_num=100,attention_window=20, downsample=True, dropout=0.5,
                 kernel_sizes=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]):
        super(BasicLayer, self).__init__()
        self.d_k = planes

        self.kernel_sizes = kernel_sizes
        self.h = len(self.kernel_sizes)
        self.d_model = self.d_k * self.h

        self.attn = None
        self.in_channels = in_channels
        self.models = nn.ModuleList([
            clones(nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=planes // 4, kernel_size=1, bias=False),

                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes // 4),

                nn.Conv1d(planes // 4, planes, kernel_size, stride, (kernel_size - 1) // 2, bias=False),

                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes),

                nn.Conv1d(planes, planes, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(planes),

                # nn.Conv2d(in_channels=in_channels,out_channels=planes,)
            ), 3) for kernel_size in self.kernel_sizes
        ])

        self.linears = clones(nn.Linear(self.d_model, self.d_model), 3)
        self.dropout_p = dropout

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # self.sublayer= SublayerConnection(self.d_model,dropout)
        self.sublayer_bn = nn.BatchNorm1d(in_channels)

        # self.sublayer_bn =LayerNorm(batch_size)
        self.sublayer_dropout = nn.Dropout(p=dropout)

        # self.attention_mask= torch.zeros(())
        self.global_index_num=global_index_num
        self.attention_window=attention_window

    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        # if self.in_channels>1:
        #     x1= self.sublayer_bn(x)
        #     # x1=self.sublayer_bn(x.permute(0,2,1)).permute(0,2,1)
        # else:
        #     x1=x
        nbatches,channels,seq_len= x.shape
        attention_mask= x.new_zeros((nbatches,seq_len))
        # global_index_num= 100
        if self.global_index_num>0:
            interval = seq_len// self.global_index_num
            attention_mask[:,::interval]=1
        x1 = self.sublayer_bn(x)


        # x1:[batch, channel, seq_len ]->  [batch,seq_len,channel] -> [batch,seq_len,h,d_model//h]
        query, key, value = [
            torch.cat([self.models[i][j](x1) for i in range(len(self.kernel_sizes))], dim=1).permute(0, 2, 1).view(
                nbatches, -1, self.h, self.d_k).permute(0, 2, 1, 3) for j in range(3)]

        # query,key,value=[torch.cat([self.bottleneck[i][j](x) for i in range(len(self.kernel_sizes))],dim=1).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for j in range(3)]
        # calculate q,k,v : [batch,h,seq_len,d_model//h]-> out:[batch,h,seq_len,d_model//h],atten:[batch,h,seq_len,seq_len]
        # out, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        out,self.attn =attention(query,key,value,mask=mask,dropout=self.dropout)
        # batch_size * h * len * hidden_dim
        # out=out.transpose(1,2).contiguous().view(nbatches,-1,self.d_model)
        # out : [batch, channel,seq_len ]
        out = out.permute(0, 1, 3, 2).contiguous().view(nbatches, self.d_model, -1)
        if x1.shape[1] == out.shape[1]:
            return x + self.sublayer_dropout(out)
        else:
            return out


class TSEncoder(nn.Module):
    def __init__(self,global_index_num=100,attention_window=20):
        super(TSEncoder, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1,global_index_num,attention_window ),
                                    BasicLayer(self.hidden*self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window))
        self.bn = nn.BatchNorm1d(self.hidden*self.kernel_num)
        self.conv1d = nn.Conv1d(self.hidden*self.kernel_num, self.hidden, 1, 1)
        self.out_bn= nn.BatchNorm1d(self.hidden)
        self.out = nn.Conv1d(self.hidden , 1, 1, 1)


    def forward(self, x):
        # x:[batch,channel,seq_len,]-> x:[batch,channel,seq_len]
        x = self.layers(x)
        # return :[batch,seq_len]
        return self.out(self.out_bn(torch.relu(self.conv1d(self.bn(x))))).squeeze()
        # return self.conv1d(x).squeeze()




class TSEncoderDTWED(nn.Module):
    def __init__(self,global_index_num=100,attention_window=20,similar_matrix=None):
        super(TSEncoderDTWED, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1,global_index_num,attention_window ),
                                    BasicLayer(self.hidden*self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1, global_index_num,
                                               attention_window))
        self.bn = nn.BatchNorm1d(self.hidden*self.kernel_num)
        self.conv1d = nn.Conv1d(self.hidden*self.kernel_num, self.hidden, 1, 1)
        self.similar_matrix= similar_matrix
        # self.out_bn= nn.BatchNorm1d(self.hidden)
        # self.out = nn.Conv1d(self.hidden , 1, 1, 1)


    def forward(self, x):
        batch_size=x.shape[0]
        # x:[batch,channel,seq_len,]-> x:[batch,channel,seq_len]
        x = self.layers(x)
        # return :[batch,hidden*seq_len]
        x=self.conv1d(self.bn(x)).reshape(batch_size,-1)
        # idx=np.random.randint(0, batch_size)
        # res=torch.cat([torch.matmul(x[i],x[]) for i in range(batch_size)],dim=0)
        res=[]
        for i in range(batch_size):
            index=torch.ones(batch_size,dtype=torch.long)
            index[i]=0
            b= x[index==1,:].transpose(-2,-1)
            a=x[i,:].reshape(1,-1)
            res.append(torch.matmul(a,b))
            # res.append(torch.softmax(torch.matmul(a,b),dim=1))
        res = torch.cat(res,dim=0)
        return res
        # return res/torch.sum(res,dim=1).unsqueeze(-1)
        # return self.out(self.out_bn(torch.relu(self.conv1d(self.bn(x))))).squeeze()
        # return self.conv1d(x).squeeze()


class TSClassifcation(nn.Module):
    def __init__(self, nclass,global_index_num=100,attention_window=20):
        super(TSClassifcation, self).__init__()
        self.inplanes = 1
        self.hidden = 32
        self.kernel_num = 12
        self.layers = nn.Sequential(BasicLayer(1, self.hidden, 1, global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window),
                                    BasicLayer(self.hidden * self.kernel_num, self.hidden, 1,global_index_num,attention_window))
        self.conv1d_new = nn.Conv1d(self.hidden * self.kernel_num, nclass, 3, 1, 1)
        self.adapt_pool = nn.AdaptiveAvgPool1d(1)
        # self.adapt_pool = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(nclass, nclass, 1, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv1d_new(x)
        # out1:[batch,nclass,seq_len]
        # out1= torch.softmax(x,dim=1)
        # out2 :[batch,nclass,1]
        out2 = self.out(self.adapt_pool(torch.relu(x)))
        return x, out2.squeeze(-1)
        # out2= torch.softmax(self.out(self.adapt_pool(torch.relu(x))),dim=1)
        # return out1,out2.squeeze()


if __name__ == '__main__':
    # batch_size=2
    # channels=1
    # seq_len =1024
    # x= torch.randn(batch_size,channels,seq_len)
    # encoder= TSEncoder()
    # device= torch.device('cuda:0')
    # x=x.to(device)
    # encoder.to(device)
    # b=encoder(x)
    # print(b.shape)
    # print(b)

    # for TSClassifcation
    batch_size = 64
    channels = 1
    seq_len = 1000
    x = torch.randn(batch_size, channels, seq_len)
    # encoder = TSEncoder()
    model = TSClassifcation(3)
    device = torch.device('cuda:0')
    x = x.to(device)
    model.to(device)

    out1, out2 = model(x)
    print(out1.shape)
    print(out2.shape)
    # print(out2)

# class Decoder(nn.Module):
#     def __init__(self,layer,N):
#         super(DecoderLayer, self).__init__()
#         self.layers= clones(layer,N)
#         self.norm= LayerNorm(layer.size)
#     def forward(self, x,memory,src_mask,tgt_mask):
#         for layer in self.layers:
#             x= layer(x,memory,src_mask,tgt_mask)
#         return self.norm(x)
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self,):
#         super(DecoderLayer, self).__init__()
#
