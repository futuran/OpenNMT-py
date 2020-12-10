""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, encoder2, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.encoder2 = encoder2
        #self.weight = torch.autograd.Variable(xxxxx)
        self.pooler = nn.AdaptiveAvgPool1d(10)
        self.decoder = decoder

    def forward(self, src, sim, tgt, src_lengths, sim_lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, src_lengths)

        enc_state2, memory_bank2, lengths2 = self.encoder(sim, sim_lengths)
        enc_pooled2 = self.pooler(enc_state2.transpose(2,0)).transpose(2,0)
        mb_pooled2 = self.pooler(memory_bank2.transpose(2,0)).transpose(2,0)

        print('1 batch...')
        print(enc_state.size())
        print(enc_state2.size())
        print(enc_pooled2.size())
        print(mb_pooled2.size())
        print(lengths.size())
        
        enc_out=torch.cat([enc_state,enc_pooled2])
        mb_out=torch.cat([memory_bank,mb_pooled2])
        print(enc_out.size())


        if bptt is False:
            self.decoder.init_state(src, mb_out, enc_out)
        dec_out, attns = self.decoder(dec_in, mb_out,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.encoder2.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
