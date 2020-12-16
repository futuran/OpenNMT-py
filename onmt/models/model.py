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
        #self.sim_weight = torch.eye(512,requires_grad=True)
        #self.weight = torch.autograd.Variable(torch.Tensor([1]),requires_grad=True)
        self.pooler = nn.AdaptiveAvgPool1d(10)
        self.decoder = decoder

        #print("model initialized!!")

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

        #print("start batch")
        #print(src.size())
        #print(sim.size())
        #print(sim.dtype)

        src_enc_out, src_memory_bank, src_lens = self.encoder(src, src_lengths)
        sim_enc_out, sim_memory_bank, sim_lens = self.encoder2(sim, sim_lengths)



        sim_pooled_enc = self.pooler(sim_enc_out.transpose(2,0)).transpose(2,0)
        sim_pooled_mb  = self.pooler(sim_memory_bank.transpose(2,0)).transpose(2,0)
        

        #print(self.sim_weight.device)
        #print(self.weight.device)
        #print(self.weight.grad)
        #sim_lineared_enc = torch.div(sim_pooled_enc, self.weight)
        #sim_lineared_mb  = torch.div(sim_pooled_mb,  self.weight)
        #sim_lineared_enc = torch.bmm(sim_pooled_enc.transpose(0,1), self.sim_weight.expand(src.size()[1],512,512).to(src.device)).transpose(0,1)
        #sim_lineared_mb  = torch.bmm(sim_pooled_mb.transpose(0,1), self.sim_weight.expand(src.size()[1],512,512).to(src.device)).transpose(0,1)

        src_out=torch.cat([torch.zeros(10,src.size()[1],src.size()[2],dtype=src.dtype, device=src.device),src])

        #print(src_enc_out.size())
        #print(sim_lineared_enc.size())
        #print(torch.norm(self.weight))

        #enc_out = torch.cat([sim_lineared_enc, src_enc_out])
        #mb_out  = torch.cat([sim_lineared_mb,  src_memory_bank])


        #src_out = torch.cat([sim,               src])
        enc_out = torch.cat([sim_pooled_enc,  src_enc_out])
        mb_out  = torch.cat([sim_pooled_mb,   src_memory_bank])


        #print(src_out.size())

        if bptt is False:
            self.decoder.init_state(src_out, mb_out, enc_out)
        dec_out, attns = self.decoder(dec_in, mb_out, memory_lengths=(src_lens+10), with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.encoder2.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
