.SUFFIXES: .uml .dot .svg

EXPECTED_UMLS = $(wildcard tests/expected/*/*.uml)
ACTUAL_UMLS = $(wildcard tests/actual/*/*.uml)

EXPECTED_DOTS = $(wildcard tests/expected/*/*.dot)
ACTUAL_DOTS = $(wildcard tests/actual/*/*.dot)

EXPECTED_DOT_SVGS = $(patsubst %.dot, %.svg, $(EXPECTED_DOTS))
ACTUAL_DOT_SVGS = $(patsubst %.dot, %.svg, $(ACTUAL_DOTS))

EXPECTED_UML_SVGS = $(patsubst %.uml, %.svg, $(EXPECTED_UMLS))
ACTUAL_UML_SVGS = $(patsubst %.uml, %.svg, $(ACTUAL_UMLS))

EXPECTED_SVGS = $(EXPECTED_DOT_SVGS) $(EXPECTED_UML_SVGS)
ACTUAL_SVGS = $(ACTUAL_DOT_SVGS) $(ACTUAL_UML_SVGS)

DOT_SVGS = $(EXPECTED_DOT_SVGS) $(ACTUAL_DOT_SVGS)
UML_SVGS = $(EXPECTED_UML_SVGS) $(ACTUAL_UML_SVGS)

ALL_SVGS = $(DOT_SVGS) $(UML_SVGS)

all: svgs

svgs: $(ALL_SVGS)

.dot.svg:
	dot -Tsvg $? > $@

$(UML_SVGS): plantuml.jar

.uml.svg:
	java -jar plantuml.jar -tsvg $<

plantuml.jar:
	wget http://sourceforge.net/projects/plantuml/files/plantuml.jar/download -O plantuml.jar

build:
	(cargo fmt && cargo build) 2>&1 | tee junk

fast:
	(cargo fmt && cargo test) 2>&1 | tee junk

slow:
	(cargo fmt && cargo test -- --test-threads 1 --nocapture) 2>&1 | tee junk

TODO = todox  # ALLOW TODOX

$(TODO):
	(cargo fmt && cargo test $(TODO) -- --test-threads 1 --nocapture) 2>&1 | tee junk
