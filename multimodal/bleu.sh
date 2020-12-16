
sed -r 's/(@@ )|(@@ ?$)//g' jaen2en/output.en > jaen2en/output.en.r
perl /home/futuran/work/OpenNMT-py/tools/multi-bleu.perl ./data/NFR_FlickrCOCO_multi/merge.0.00/flickr_test.en1.tokenized.with_match.bpe.r < jaen2en/output.en.r > out.bleu
