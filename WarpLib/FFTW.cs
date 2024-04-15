using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Security;
using System.Runtime.InteropServices;

namespace Warp
{
	[SuppressUnmanagedCodeSecurity]
    public static class FFTW
    {
        [DllImport("fftw3f", EntryPoint = "fftwf_init_threads")]
        public static extern bool init_threads();


		[DllImport("fftw3f", EntryPoint = "fftwf_plan_with_nthreads")]
		public static extern void plan_with_nthreads(int nthreads);


        [DllImport("fftw3f", EntryPoint = "fftwf_plan_dft")]
        public static extern IntPtr plan_dft(int rank, [MarshalAs(UnmanagedType.LPArray)] int[] n, IntPtr arrIn, IntPtr arrOut, DftDirection direction, PlannerFlags flags);


        [DllImport("fftw3f", EntryPoint = "fftwf_plan_dft_r2c")]
        public static extern IntPtr plan_dft_r2c(int rank, [MarshalAs(UnmanagedType.LPArray)] int[] n, IntPtr arrIn, IntPtr arrOut, PlannerFlags flags);


        [DllImport("fftw3f", EntryPoint = "fftwf_plan_dft_c2r")]
        public static extern IntPtr plan_dft_c2r(int rank, [MarshalAs(UnmanagedType.LPArray)] int[] n, IntPtr arrIn, IntPtr arrOut, PlannerFlags flags);


        [DllImport("fftw3f", EntryPoint = "fftwf_destroy_plan")]
        public static extern void destroy_plan(IntPtr plan);


        [DllImport("fftw3f", EntryPoint = "fftwf_execute")]
        public static extern void execute(IntPtr plan);


        [DllImport("fftw3f", EntryPoint = "fftwf_alloc_real")]
        public static extern IntPtr alloc_real(long size);


        [DllImport("fftw3f", EntryPoint = "fftwf_alloc_complex")]
        public static extern IntPtr alloc_complex(long size);


        [DllImport("fftw3f", EntryPoint = "fftwf_free")]
        public static extern void free(IntPtr ptr);
    }

    public enum DftDirection : int
    {
        Forwards = -1,
        Backwards = 1
    }

    [Flags]
    public enum PlannerFlags : uint
    {
        Default = Measure,

        /// <summary>
        /// FFTW_MEASURE tells FFTW to find an optimized plan by actually computing several FFTs 
        /// and measuring their execution time. Depending on your machine, this can take some time 
        /// (often a few seconds). FFTW_MEASURE is the default planning option.
        /// </summary>
        Measure = (0U),

        /// <summary>
        /// FFTW_EXHAUSTIVE is like FFTW_PATIENT, but considers an even wider range of algorithms, 
        /// including many that we think are unlikely to be fast, to produce the most optimal 
        /// plan but with a substantially increased planning time.
        /// </summary>
        Exhaustive = (1U << 3),

        /// <summary>
        /// FFTW_PATIENT is like FFTW_MEASURE, but considers a wider range of algorithms and 
        /// often produces a “more optimal” plan (especially for large transforms), but at the 
        /// expense of several times longer planning time (especially for large transforms).
        /// </summary>
        Patient = (1U << 5),

        /// <summary>
        /// FFTW_ESTIMATE specifies that, instead of actual measurements of different algorithms, 
        /// a simple heuristic is used to pick a (probably sub-optimal) plan quickly. With this 
        /// flag, the input/output arrays are not overwritten during planning.
        /// </summary>
        Estimate = (1U << 6),

        /// <summary>
        /// FFTW_WISDOM_ONLY is a special planning mode in which the plan is only created 
        /// if wisdom is available for the given problem, and otherwise a NULL plan is returned. 
        /// This can be combined with other flags, e.g. ‘FFTW_WISDOM_ONLY | FFTW_PATIENT’ 
        /// creates a plan only if wisdom is available that was created in FFTW_PATIENT or 
        /// FFTW_EXHAUSTIVE mode. The FFTW_WISDOM_ONLY flag is intended for users who need to 
        /// detect whether wisdom is available; for example, if wisdom is not available one may 
        /// wish to allocate new arrays for planning so that user data is not overwritten.
        /// </summary>
        WisdomOnly = (1U << 21)
    }
}
