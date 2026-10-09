using System;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Independent hypothesis groups; within each group spectral envelopes
/// interpolate between the negative endpoint, zero, and positive endpoint.
/// Patch amplitudes remain free. SPA uses one common envelope shape.</summary>
public sealed record CtfEnvelopeLayout(int Groups, int Anchors, int[] Ids, double[] Blends)
{
    public static CtfEnvelopeLayout Create(int count, int[] groups = null, double[] angles = null)
    {
        if(count<1 || (groups!=null&&groups.Length!=count) || (angles!=null&&angles.Length!=count))
            throw new ArgumentException("Invalid CTF envelope layout.");
        groups??=new int[count];angles??=new double[count];
        if(angles.Any(x=>!double.IsFinite(x)))throw new ArgumentException("Nonfinite envelope tilt angle.");
        var labels=groups.Distinct().OrderBy(x=>x).ToArray();
        var ids=groups.Select(g=>Array.IndexOf(labels,g)).ToArray();
        bool varying=Enumerable.Range(0,labels.Length).Any(g=>angles.Where((_,i)=>ids[i]==g).Distinct().Count()>1);
        int anchors=varying?3:1;var blends=new double[count*anchors];
        for(int g=0;g<labels.Length;g++)
        {
            var members=Enumerable.Range(0,count).Where(i=>ids[i]==g).ToArray();
            double lo=Math.Min(0,members.Min(i=>angles[i])),hi=Math.Max(0,members.Max(i=>angles[i]));
            foreach(int i in members)
            {
                if(anchors==1){blends[i]=1;continue;}
                double negative=lo<0?Math.Max(0,angles[i]/lo):0,positive=hi>0?Math.Max(0,angles[i]/hi):0;
                blends[3*i]=negative;blends[3*i+1]=1-negative-positive;blends[3*i+2]=positive;
            }
        }
        return new(labels.Length,anchors,ids,blends);
    }
    public static double[] Angles(CtfFitGeometry[] geometry) => geometry.Select(g=>g.Rotation.HasValue?
        Math.Atan2(-g.Rotation.Value.M31,g.Rotation.Value.M33)*180/Math.PI:0).ToArray();
}
