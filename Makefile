SUBDIRS = componentes_compartidos secuencial open_mp

.PHONY: $(SUBDIRS) all testall exeall liball cleanall \
		componentes_compartidos
		
all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir all; \
	done

testall:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done

exeall:
	for dir in $(SUBDIRS); do \
		if [$$dir -neq 'componentes_compartidos'] \
		then \
			$(MAKE) -C $$dir exe; \
		fi\
	done

liball:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir lib; \
	done

cleanall: 
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir cleanall; \
	done