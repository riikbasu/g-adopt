include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := mantle_convection
params.log : ncpus := 4
params.log: 2d_compressible_TALA.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
