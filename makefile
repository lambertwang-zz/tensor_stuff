FLAGS = -Wall -Wextra -g

SRC_LOC = src/
TEST_LOC = test/
MAIN_NAME = main
EXE_NAME = tensor

INCLUDE = -I$(SRC_LOC)
LINK = -lm

CC = g++ $(FLAGS) $(INCLUDE)
EXT = .cpp

# Do not modify anything below this line

CLEAR = clear
RM = rm -rf
SRC = $(shell find src -name '*$(EXT)' ! -name '*$(MAIN_NAME)$(EXT)')
RM_OUT = 
ifeq ($(OS),Windows_NT)
# CLEAR = cls
# RM = del /s
# RM_OUT = 2>NUL
#SRC = $(shell powershell "gci src -r -i *$(EXT) | ? Name -ne "$(MAIN_NAME)$(EXT)" | resolve-path -r")
EXECUTABLE = $(EXE_NAME).exe
else
CLEAR = clear
RM = rm -rf
RM_OUT = 
SRC = $(shell find src -name '*$(EXT)' ! -name '*$(MAIN_NAME)$(EXT)')
EXECUTABLE = $(EXE_NAME)
endif

SRC_OBJ = $(SRC:$(EXT)=.o)

TEST_EXE = $(TEST:$(EXT)=)

all: clear clean_exe $(MAIN_NAME)

$(MAIN_NAME): $(SRC_OBJ)
	$(CC) $(SRC_LOC)$(MAIN_NAME)$(EXT) -o $(EXECUTABLE) $(SRC_OBJ) $(LINK)

$(EXT).o:
	$(CC) -c $< -o $@

clear:
	$(CLEAR)

clean_exe:
	$(RM) $(EXECUTABLE) $(RM_OUT)

clean:
	$(RM) $(EXECUTABLE) $(SRC_OBJ) $(RM_OUT)

