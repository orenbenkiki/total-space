.SUFFIXES: .uml .dot .svg

EXPECTED_UMLS = $(wildcard tests/expected/*/*.uml)
ACTUAL_UMLS = $(wildcard tests/actual/*/*.uml)

EXPECTED_DOTS = $(wildcard tests/expected/*/*.dot)
ACTUAL_DOTS = $(wildcard tests/actual/*/*.dot)

EXPECTED_SVGS = \
	$(patsubst %.dot, %.svg, $(EXPECTED_DOTS)) \
	$(patsubst %.uml, %.svg, $(EXPECTED_UMLS))
ACTUAL_SVGS = \
	$(patsubst %.dot, %.svg, $(ACTUAL_DOTS)) \
       	$(patsubst %.uml, %.svg, $(ACTUAL_UMLS))

all: expected_svgs actual_svgs

expected_svgs: $(EXPECTED_SVGS)

actual_svgs: $(ACTUAL_SVGS)

.dot.svg:
	dot -Tsvg $? > $@

.uml.svg: plantuml.jar
	java -jar plantuml.jar -tsvg $?

plantuml.jar:
	wget http://sourceforge.net/projects/plantuml/files/plantuml.jar/download -O plantuml.jar
