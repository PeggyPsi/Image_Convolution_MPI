#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// mpiexec -n 4 path/to/image image_type image_width image_height reps
// example: mpiexec -n 4 ./mpi_project parallel.txt grey 5 4 10

#define FILTER_ELEM_NUM 9

typedef enum {RGB, GREY} colour_model;
typedef enum {BOX_BLUR, GAUSSIAN_BLUR} filter_type_t;
typedef float* norm_filter_t;                //normalized filter
typedef int filter_t[3][3];

static inline void Convolute_Block(unsigned char * from, unsigned char * to, norm_filter_t h_filter, int start_offset, int count, int row_length, int blocklength, int pixel_size);
static inline void Convolute_Pixel(unsigned char * from, unsigned char * to, norm_filter_t h_filter, int coordx, int coordy, int row_length, int pixel_size);

static inline  int Check_Arguments(char** argv, int argc, colour_model * image_type, int * width, int * height, int *reps);
static inline void Set_Filter(int my_rank, filter_type_t type, norm_filter_t* h_filter);
static inline int Find_Neighbour(MPI_Comm *grid_comm, int *coord, int *n_rank, int * dimension_size);

int main(int argc, char** argv){

    int my_rank, numof_processes;

    /*virtual topology parameters*/
    int numof_dims, reorder, periods[1], dimension_size[1], grid_rank, coord[1];
    MPI_Comm grid_comm;

    /*virtual topology neighbours*/
    int *neighbour_ranks = malloc(9*(sizeof(int))); /*(1,1) is empty*/
    int neighbour_coord[1];

    /*main arguments*/
    int image_width, image_height, reps;
    colour_model image_type;
    MPI_File fh, fh2;
    char* image_name;

    /*Parallel IO params*/
    int err_code, global_sizes[1], local_sizes[1];
    int distirbs[1], dargs[1], numof_pixels;
    int pixel_size;
    unsigned char *local_buf, *temp_buf, *tmp;
    MPI_Datatype filetype;
    MPI_Datatype PIXEL;   //Is set to 3b for rgb, 1b for grey
    MPI_Status status;

    /*Datatypes*/
    MPI_Datatype PIXELS_BLOCK;
    MPI_Datatype row_type, column_type;

    /*Filters*/
    norm_filter_t h_filter = NULL;
    /*Time*/
    double my_time, global_time;

    //--Start MPI
    MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);                     /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &numof_processes); /* get # procs from env */


    //--Set Virtual Cartesian Topology
    numof_dims = 2;
    periods[0] = 0;
    periods[1] = 0;                     /*non periodic rows and collumns*/
    reorder = 1;                        /*allows processes to reorder*/
    dimension_size[0] = 0; /*dimensions of process grid*/
    dimension_size[1] = 0;
    MPI_Dims_create(numof_processes, 2, dimension_size);
    if(MPI_Cart_create(MPI_COMM_WORLD, numof_dims, dimension_size, periods, reorder, &grid_comm) != MPI_SUCCESS){
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    };
    MPI_Comm_rank(grid_comm, &my_rank);
    MPI_Cart_coords(grid_comm, my_rank, numof_dims, coord);


    //--Check Arguments and Open File
    image_name = malloc((strlen(argv[1])+1)*sizeof(char));
    strcpy(image_name, argv[1]);
    if(my_rank==0){
        if(Check_Arguments(argv, argc, &image_type, &image_width, &image_height, &reps)){
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          return EXIT_FAILURE;
        }
    };
    err_code = MPI_File_open(grid_comm, image_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh );
    if(err_code){
        printf("Unable to open file\n");fflush(stdout);
        MPI_File_close(&fh);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        free(image_name); image_name = NULL;
        return EXIT_FAILURE;
    };

    //--Broadcast Arguments
    MPI_Bcast(&image_width, 1, MPI_INT, 0, grid_comm);
    MPI_Bcast(&image_height, 1, MPI_INT, 0, grid_comm);
    MPI_Bcast(&reps, 1, MPI_INT, 0, grid_comm);
    MPI_Bcast(&image_type, 1, MPI_INT, 0, grid_comm);
    MPI_Barrier(grid_comm);

    //--Parallel Read
    global_sizes[0] = image_height; /* no. of rows in global array */
    global_sizes[1] = image_width; /* no. of columns in global array*/
    local_sizes[0] = global_sizes[0]/dimension_size[0];/* no. of rows in local array */
    local_sizes[1] = global_sizes[1]/dimension_size[1];/* no. of columns in local array*/
    /*if process is edge process - might need extra space*/
    if(coord[0]==0){ local_sizes[0] += global_sizes[0]%dimension_size[1];}
    if(coord[1]==0){ local_sizes[1] += global_sizes[1]%dimension_size[0];}
    distirbs[0] = MPI_DISTRIBUTE_BLOCK;
    distirbs[1] = MPI_DISTRIBUTE_BLOCK;
    dargs[0] = MPI_DISTRIBUTE_DFLT_DARG;
    dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;

    if(image_type == RGB){
        MPI_Type_contiguous(3, MPI_BYTE, &PIXEL);
    }
    else{
        MPI_Type_contiguous(1, MPI_BYTE, &PIXEL);
    }
    MPI_Type_commit(&PIXEL);
    MPI_Type_size(PIXEL, &pixel_size);
    MPI_Type_vector(local_sizes[0], local_sizes[1], local_sizes[1]+2, PIXEL, &PIXELS_BLOCK);
    MPI_Type_commit(&PIXELS_BLOCK);

    MPI_Type_create_darray(numof_processes, my_rank, 2, global_sizes, distirbs, dargs, dimension_size, MPI_ORDER_C,  PIXEL, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);

    numof_pixels = (local_sizes[0]+2)*(local_sizes[1]+2);       /*size of local buf in pixels with hollow points*/
    local_buf = (unsigned char*)malloc(numof_pixels*pixel_size); /*local part of file*/
    memset(local_buf, 10, numof_pixels*pixel_size);
    MPI_File_read_all(fh, &local_buf[(local_sizes[1]+2)*pixel_size +pixel_size], 1, PIXELS_BLOCK, &status);
    MPI_File_close(&fh);

    // if(my_rank == 0){
    //   printf("Block of pixels and hollow points of process %d:\n", my_rank);
    //   for(int i=0; i<local_sizes[0]+2; i++){
    //     for(int j=0; j<(local_sizes[1]+2)*pixel_size; j++){
    //       printf("%d ", local_buf[i*((local_sizes[1]+2)*pixel_size) + j]);
    //       // if(j == local_sizes[1]-1) printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    //--Find Neighbour Processes
    for(int i=0; i<=2; i++){
        for(int j=0; j<=2; j++){
          if(i == 1 && j == 1) {continue;}
          neighbour_coord[0] = coord[0] +(i-1);
          neighbour_coord[1] = coord[1] +(j-1);
          Find_Neighbour(&grid_comm, neighbour_coord, (neighbour_ranks +i*3 +j), dimension_size);
        }
    }

    //--Set filters
    h_filter = (norm_filter_t)malloc(FILTER_ELEM_NUM*sizeof(float));
    // Set_Filter(my_rank, BOX_BLUR, &h_filter);
    Set_Filter(my_rank, GAUSSIAN_BLUR, &h_filter);

    //--Set Communication Datatypes
    MPI_Type_contiguous(local_sizes[1], PIXEL, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(local_sizes[0], 1, local_sizes[1]+2, PIXEL, &column_type);
    MPI_Type_commit(&column_type);

    MPI_Barrier(grid_comm);
    my_time = MPI_Wtime();

    MPI_Request send_requests[8];
    MPI_Status send_statuses[8];
    MPI_Request receive_requests[4];
    MPI_Request corner_receive_requests[4];

    temp_buf = (unsigned char*)malloc(numof_pixels*pixel_size); /*local part of file*/
    memset(temp_buf, 10, numof_pixels*pixel_size);

    //--Actual convolution
    int diff; // for checking whether convolution has no longer effect on the image for current process
    for (int n=0; n<reps; n++){

        int offset = (local_sizes[1]+2)*pixel_size + pixel_size;
        //--Send Requests
        MPI_Isend(&local_buf[offset], 1, PIXEL, neighbour_ranks[0], 0, grid_comm, &send_requests[0]);
        MPI_Isend(&local_buf[offset], 1, row_type, neighbour_ranks[1], 0, grid_comm, &send_requests[1]);
        MPI_Isend(&local_buf[offset +(local_sizes[1]-1)*pixel_size], 1, PIXEL, neighbour_ranks[2], 0, grid_comm, &send_requests[2]);

        MPI_Isend(&local_buf[offset], 1, column_type, neighbour_ranks[3], 0, grid_comm, &send_requests[3]);
        MPI_Isend(&local_buf[offset +(local_sizes[1]-1)*pixel_size], 1, column_type, neighbour_ranks[5], 0, grid_comm, &send_requests[4]);

        offset = local_sizes[0]*(local_sizes[1]+2)*pixel_size + pixel_size;
        MPI_Isend(&local_buf[offset], 1, PIXEL, neighbour_ranks[6], 0, grid_comm, &send_requests[5]);
        MPI_Isend(&local_buf[offset], 1, row_type, neighbour_ranks[7], 0, grid_comm, &send_requests[6]);
        MPI_Isend(&local_buf[offset +(local_sizes[1]-1)*pixel_size], 1, PIXEL, neighbour_ranks[8], 0, grid_comm, &send_requests[7]);

        //--Receive Requests
        offset = 0;
        MPI_Irecv(&local_buf[offset], 1, PIXEL, neighbour_ranks[0], 0, grid_comm, &corner_receive_requests[0]);
        offset = pixel_size;
        MPI_Irecv(&local_buf[offset], 1, row_type, neighbour_ranks[1], 0, grid_comm, &receive_requests[0]);
        offset = (local_sizes[1]+1)*pixel_size;
        MPI_Irecv(&local_buf[offset], 1, PIXEL, neighbour_ranks[2], 0, grid_comm, &corner_receive_requests[1]);

        offset = (local_sizes[1]+2)*pixel_size;
        MPI_Irecv(&local_buf[offset], 1, column_type, neighbour_ranks[3], 0, grid_comm, &receive_requests[1]);
        offset = (local_sizes[1]+2)*pixel_size + pixel_size*(local_sizes[1]+1);
        MPI_Irecv(&local_buf[offset], 1, column_type, neighbour_ranks[5], 0, grid_comm, &receive_requests[2]);
        // offset += pixel_size;

        offset = ((local_sizes[1]+2)*pixel_size)*(local_sizes[0]+1);
        MPI_Irecv(&local_buf[offset], 1, PIXEL, neighbour_ranks[6], 0, grid_comm, &corner_receive_requests[2]);
        offset += pixel_size;
        MPI_Irecv(&local_buf[offset], 1, row_type, neighbour_ranks[7], 0, grid_comm, &receive_requests[3]);
        offset += local_sizes[1]*pixel_size;
        MPI_Irecv(&local_buf[offset], 1, PIXEL, neighbour_ranks[8], 0, grid_comm, &corner_receive_requests[3]);

        //--Convolute Inner Pixels
        Convolute_Block(&local_buf[(local_sizes[1]+2)*2*pixel_size], &temp_buf[(local_sizes[1]+2)*2*pixel_size], h_filter,  2, local_sizes[0]-2, local_sizes[1]+2, local_sizes[1]-2, pixel_size);

        //--Convolute Outer Pixels

        int index=0, flag;
        /*borders*/
        while(1){
            err_code = MPI_Testany(4, receive_requests, &index, &flag, &status);
            if(!flag){
                err_code = MPI_Waitany(4, receive_requests, &index, &status);
            }

            /*convolute what got received*/
            int starting_point; /*points to the start of the row/column*/
            /*up or down border*/
            if(index==MPI_UNDEFINED) break;
            else if(index==0){
              starting_point = ((local_sizes[1]+2)*pixel_size);
              Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, 2, 1, local_sizes[1]+2, local_sizes[1]-2, pixel_size);
            }
            else if(index==3){
              starting_point = ((local_sizes[1]+2)*pixel_size)*local_sizes[0];
              Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, 2, 1, local_sizes[1]+2, local_sizes[1]-2, pixel_size);

            }
            /*left or right border*/
            else if(index==1){
              starting_point = (local_sizes[1]+2)*pixel_size*2;
              Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, 1, local_sizes[0]-2, local_sizes[1]+2, 1, pixel_size);
            }
            else if(index==2){
              starting_point = (local_sizes[1]+2)*pixel_size*2;
              Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, local_sizes[1], local_sizes[0]-2, local_sizes[1]+2, 1, pixel_size);

            }
        }

        /*corners*/
        index = 0; flag = 0;
        while(1){
            err_code = MPI_Testany(4, corner_receive_requests, &index, &flag, &status);
            if(!flag){
                err_code = MPI_Waitany(4, corner_receive_requests, &index, &status);
            }
            /*convolute what got received*/
            int starting_point; /*points to the start of the row/column*/
            if(index==MPI_UNDEFINED) break;
            else if(index==0){
                starting_point = (local_sizes[1]+2)*pixel_size;
                Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, 1, 1, local_sizes[1]+2, 1, pixel_size);
            }
            else if(index==1){
                starting_point = (local_sizes[1]+2)*pixel_size;
                Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, local_sizes[1], 1, local_sizes[1]+2, 1, pixel_size);
            }
            else if(index==2){
                starting_point = ((local_sizes[1]+2)*pixel_size)*local_sizes[0];
                Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, 1, 1, local_sizes[1]+2, 1, pixel_size);
            }
            else if(index==3){
                starting_point = ((local_sizes[1]+2)*pixel_size)*local_sizes[0];
                Convolute_Block(&local_buf[starting_point], &temp_buf[starting_point], h_filter, local_sizes[1], 1, local_sizes[1]+2, 1, pixel_size);
            }
        }

        //--Wait Send Requests to Complete
        err_code = MPI_Waitall(8, send_requests, send_statuses);

        tmp = local_buf;
        local_buf = temp_buf;
        temp_buf = tmp;

        diff = 0;
        if((n+1)%10==0){
            for(int i=1*(local_sizes[1]+2)*pixel_size + pixel_size ; i<local_sizes[0]*(local_sizes[1]+2)*pixel_size + pixel_size*(local_sizes[1]+1) ; i++){
                if(local_buf[i] != temp_buf[i]) diff = 1;
                break;
            }
            if(diff == 0) break;
        }

        // if(my_rank == 0){
        //   printf("Block of pixels and hollow points of process %d:\n", my_rank);
        //   for(int i=0; i<local_sizes[0]+2; i++){
        //     for(int j=0; j<(local_sizes[1]+2)*pixel_size; j++){
        //       printf("%d ", temp_buf[i*((local_sizes[1]+2)*pixel_size) + j]);
        //       // if(j == local_sizes[1]-1) printf("\n");
        //     }
        //     printf("\n");
        //   }
        // }

    }

    //naem new convoluted image
    char* new_image_name = malloc(strlen(image_name)+7+1);
    memcpy(new_image_name, image_name, strlen(image_name)-4+1);
    memcpy(new_image_name+strlen(image_name)-4, "_conved.raw", 12);
    new_image_name[strlen(new_image_name)] = '\0';
    //create, open and store convoluted pixels in new image
    MPI_File_open(grid_comm, new_image_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh2 );
    MPI_File_set_view(fh2, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh2, &local_buf[(local_sizes[1]+2)*pixel_size +pixel_size], 1, PIXELS_BLOCK, &status);
    MPI_File_close(&fh2);

    // MPI_Barrier(grid_comm);
    my_time = MPI_Wtime() - my_time;
    MPI_Reduce(&my_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);

    MPI_Type_free(&filetype);
    MPI_Type_free(&PIXEL);
    MPI_Type_free(&PIXELS_BLOCK);
    MPI_Type_free(&column_type);
    MPI_Type_free(&row_type);
    free(local_buf); local_buf = NULL;
    free(temp_buf); temp_buf = NULL;
    free(image_name); image_name = NULL;
    free(new_image_name); new_image_name = NULL;
    MPI_Finalize();
}

//--FUNCTIONS

static inline int Find_Neighbour(MPI_Comm *grid_comm, int *coord, int *n_rank, int * dimension_size){
  if((coord[0]==-1 || coord[0]==dimension_size[0]) || (coord[1]==-1 || coord[1]==dimension_size[1])){/*neighbour doesn't exist*/
    *n_rank = MPI_PROC_NULL;
    return 0;
  };
  if(MPI_Cart_rank(*grid_comm, coord, n_rank)==MPI_SUCCESS){
    return 0;
  };
  fprintf(stderr, "Error: problem with finding neighbour\n");
  return 1;
}

static inline void Convolute_Block(unsigned char * from, unsigned char * to, norm_filter_t h_filter, int start_offset, int count, int row_length, int blocklength, int pixel_size){

  for(int i=0; i<count; i++){
    for(int j=start_offset; j<blocklength+start_offset; j++){
      // from[i*((row_length)*pixel_size) + j*pixel_size] is starting position of pixel
      Convolute_Pixel(from, to, h_filter, i, j, row_length, pixel_size);

    }
  }
}

static inline void Convolute_Pixel(unsigned char * from, unsigned char * to, norm_filter_t h_filter, int coordx, int coordy, int row_length, int pixel_size){
  int pos;
  if(pixel_size==1){
    float grey = 0;
    for(int i = 0; i < 3; i++ ){
      for(int j = 0; j < 3; j++){
        pos = (coordx+i-1)*row_length +(coordy+j-1);
        grey += from[pos]*h_filter[i*3 +j];
      }
    }
    pos = (coordx)*row_length +(coordy);
    to[pos] = grey;
  }
  else if(pixel_size==3){
    float red = 0, green = 0, blue = 0;
    for(int i = 0; i < 3; i++ ){
      for(int j = 0; j < 3; j++){
        pos = (coordx+i-1)*row_length*pixel_size +(coordy+j-1)*pixel_size;
        red += from[pos]*h_filter[i*3 +j];
        green += from[pos+1]*h_filter[i*3 +j];
        blue += from[pos+2]*h_filter[i*3 +j];

        // printf("%c %c %c\n",from[pos],from[pos+1],from[pos+2]);
      }
    }
    pos = (coordx)*row_length*pixel_size +(coordy)*pixel_size;
    to[pos] = red;
    to[pos+1] = green;
    to[pos+2] = blue;
  }
}

static inline int Check_Arguments(char** argv, int argc, colour_model * image_type, int * width, int * height, int *reps){
    //check if program is provided with correct arguments

    if(argc == 6){
        if(!strcmp(argv[2], "grey")){
          *image_type = GREY;
        }
        else if(!strcmp(argv[2], "rgb")){
          *image_type = RGB;
        }
        else{
            fprintf(stderr, "Error: wrong colour model input\n");
            return 1;
        };
        *width = atoi(argv[3]);
        *height = atoi(argv[4]);
        *reps = atoi(argv[5]);
    }
    else{
        fprintf(stderr, "Error: wrong arguments were provided\n");
        return 1;
    }
    return 0;
}

static inline void Set_Filter(int my_rank, filter_type_t type, norm_filter_t* h_filter){
    if(type == BOX_BLUR){
        filter_t flt = {{1, 2, 1}, {2, 1, 2}, {1, 2, 1}};
        //normalize filter
        int i,j;
        for(i=0 ; i<3 ; i++){
            for(j=0 ; j<3 ; j++){
                // if(my_rank == 0) printf("%d%d ", i, j);
                (*h_filter)[i*3+j] = flt[i][j] / 16.0;
            }
        }
    }
    else if(type == GAUSSIAN_BLUR){
        filter_t flt = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        //normalize filter
        for(int i=0 ; i<3 ; i++){
            for(int j=0 ; j<3 ; j++){
                (*h_filter)[i*3+j] = flt[i][j] / 9.0;
            }
        }
    }
}
