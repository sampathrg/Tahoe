SRC = main.cu
OBJ = $(SRC:.cu=.o)
 
all : Tahoe
 
Tahoe: $(SRC)
	nvcc -I./cub-1.8.0 -O3 --expt-extended-lambda -std=c++11 -o $@ $(SRC)
	#nvcc -g -G -I/home/cc/Fast_Tree_Inference/cub-1.8.0 -O3 --expt-extended-lambda -std=c++11 -o $@ $(SRC)

clean:
	echo "[clean]"
	rm -f Tahoe
