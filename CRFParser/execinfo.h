//#include <features.h>

//__BEGIN_DECLS

/* Store up to SIZE return address of the current program state in
   ARRAY and return the exact number of values stored.  */
int backtrace(void **__array, int __size)  { return -1; };


/* Return names of functions from the backtrace list in ARRAY in a newly
   malloc()ed memory block.  */
char **backtrace_symbols (void  *__array, int __size) { return 0; };


/* This function is similar to backtrace_symbols() but it writes the result
   immediately to a file.  */
void backtrace_symbols_fd (void  *__array, int __size, int __fd)  {  };;

//__END_DECLS