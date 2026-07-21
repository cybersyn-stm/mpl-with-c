src := $(shell find -name *.c)
target := main

all : compile
	@./$(target)
compile :
	@gcc -Wall -Wextra -ggdb -o $(target) $(src) -lm
gif :
	@./make-gif.sh
clean :
	@rm -f main

