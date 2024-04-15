/* src/config.h.  Generated from config.h.in by configure.  */
/* src/config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
/* #undef F77_FUNC */

/* As F77_FUNC, but for C identifiers containing underscores. */
/* #undef F77_FUNC_ */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define to 1 if you have the `clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if C supports variable-length arrays. */
//#define HAVE_C_VARARRAYS 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* FFTW3 library is available */
/* #undef HAVE_FFTW3 */

/* FFTW3F library is available */
/* #undef HAVE_FFTW3F */

/* Define to 1 if you have the `floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Support mmx instructions */
#define HAVE_MMX 

/* Define to 1 if you have the `posix_memalign' function. */
#define HAVE_POSIX_MEMALIGN 1

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Support SSE (Streaming SIMD Extensions) instructions */
//#define HAVE_SSE 

/* Support SSE2 (Streaming SIMD Extensions 2) instructions */
//#define HAVE_SSE2 

/* Support SSE3 (Streaming SIMD Extensions 3) instructions */
//#define HAVE_SSE3 

/* Support SSE4.1 (Streaming SIMD Extensions 4.1) instructions */
//#define HAVE_SSE4_1 

/* Support SSE4.2 (Streaming SIMD Extensions 4.2) instructions */
/* #undef HAVE_SSE4_2 */

/* Support SSSE3 (Supplemental Streaming SIMD Extensions 3) instructions */
//#define HAVE_SSSE3 

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strtol' function. */
#define HAVE_STRTOL 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* xmmintrin.h is available. */
#define HAVE_XMMINTRIN_H 1

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Use double-precision to solve for single-precision splines */
/* #undef HIGH_PRECISION */

/* Name of package */
#define PACKAGE "einspline"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "esler@uiuc.edu"

/* Define to the full name of this package. */
#define PACKAGE_NAME "einspline"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "einspline 0.9.2"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "einspline"

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.9.2"

/* Prefetch loop lead distance */
/* #undef PREFETCH_AHEAD */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Use SSE prefetch */
/* #undef USE_PREFETCH */

/* Version number of package */
#define VERSION "0.9.2"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to empty if the C99 keyword for C++ does not work. */
#define restrict

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
