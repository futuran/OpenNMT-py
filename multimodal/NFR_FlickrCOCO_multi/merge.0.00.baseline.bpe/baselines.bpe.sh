for tvt in train val test; do
    subword-nmt apply-bpe -c ../tamura_preprocess/subword_3k/codes.ja < ../merge.0.00.restore/flickr_${tvt}.ja1.tokenized.with_match.bpe.jaonly.r > flickr_${tvt}.ja1.tokenized.with_match.bpe.jaonly.r.blbpe
done
