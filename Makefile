# Can also be gpu
COMPUTE=cpu

install:
	@pip install -e .[${COMPUTE}]

test:
	@letters save-datasets config/test.yml
	@letters train-model config/test.yml

pipeline:
	@letters save-datasets config/pipeline.yml
	@letters train-model config/pipeline.yml

deploy:
	@aws s3 sync ~/.mlpipes/ascii_letter/ s3://mlpipes/ascii_letter/ --exclude "*" --include "*.h5"
