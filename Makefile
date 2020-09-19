https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/train_data.zip
https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/test_data_A.zip
https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/test_data_B.zip


.PHONY: all clean load preprocess # train inference 

PYTHON = python3

all: load #results/solution/solution.csv

preprocess: data/interim/folds.csv 

# Load datasets
data/raw :load

	mkdir data/raw data/interim -p
	# Train
	wget \
		https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/train_data.zip \
		-p data/raw/
	unzip \
		-q data/raw/train_data.zip\
		-d data/raw
	rm data/raw/train_data.zip

	# Test A
	wget \
		https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/test_data_A.zip \
		-p data/raw/
	unzip \
		-q data/raw/test_data_A.zip\
		-d data/raw
	rm data/raw/test_data_A.zip	

	# Test B
	wget \
		https://digix-algo-challenge-sg.obs.ap-southeast-3.myhuaweicloud.com/2020/cv/6rKDTsB6sX8A1O2DA2IAq7TgHPdSPxJF/test_data_A.zip \
		-p data/raw/
	unzip \
		-q data/raw/test_data_B.zip\
		-d data/raw
	rm data/raw/test_data_B.zip	


data/interim/folds.csv : src/data/preprocess.py
	$(PYTHON) $< \
		--root data/raw \
		--output_path data/interim \
		--resize_images \
		--size 512 \
		--split_into_folds \
		--num_folds 5 \
		--merge_datasets
#		--resize_masks
#		# --use_color_constancy \

# # Delete everything except raw data and code
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete
# 	rm -r data/processed
# 	rm -r logs/
# 	rm -r models/
# 	rm -r results/
c2 python3 predict.py -c logs/genet_small_384_light_arcface80_1 --extract_embeddings --val_size 512 --validation --test --dba --aqe