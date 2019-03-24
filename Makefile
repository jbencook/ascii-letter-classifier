# Can also be gpu
COMPUTE=cpu

install:
	@pip install -e .[${COMPUTE}]

test:
	@letters save-datasets config/test.yml
