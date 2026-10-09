using System;
using Warp.Tools;
namespace Tests;
// Adapt zero-depth fixtures to the seven-parameter GPU model. No production fallback.
internal sealed class CtfThinGpuBatch : IDisposable
{
    readonly CtfGpuFitBatch batch;
    public CtfThinGpuBatch(CtfSpectrumFit[] spectra) { batch = new(spectra); }
    public double[] Evaluate(double[] poses, bool reweight = false)
    {
        int n=poses.Length/4; var full=new double[n*7];
        for(int i=0;i<n;i++) Array.Copy(poses,i*4,full,i*7,4);
        var output=batch.Evaluate(full,reweight); var thin=new double[n*6];
        for(int i=0;i<n;i++){Array.Copy(output,i*9,thin,i*6,5);thin[i*6+5]=output[i*9+8];}
        return thin;
    }
    public double[] Search(double[] trials,double[] offsets) => batch.Search(trials,offsets);
    public void SynchronizeWeights() => batch.SynchronizeWeights();
    public float[] ReadCoefficients() => batch.ReadCoefficients();
    public void Dispose() => batch.Dispose();
}
