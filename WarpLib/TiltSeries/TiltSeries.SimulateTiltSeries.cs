using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public Image SimulateTiltSeries(TomoProcessingOptionsBase options, int3 stackDimensions, float3[][] particleOrigins, float3[][] particleAngles, int[] nParticles, Projector[] references)
    {
        VolumeDimensionsPhysical = options.DimensionsPhysical;
        float BinnedPixelSize = (float)options.BinnedPixelSizeMean;

        Image SimulatedStack = new Image(stackDimensions);

        // Extract images, mask and resize them, create CTFs

        for (int iref = 0; iref < references.Length; iref++)
        {
            int Size = references[iref].Dims.X;
            int3 Dims = new int3(Size);

            Image CTFCoords = CTF.GetCTFCoords(Size, Size);

            #region For each particle, create CTFs and projections, and insert them into the simulated tilt series

            for (int p = 0; p < nParticles[iref]; p++)
            {
                float3 ParticleCoords = particleOrigins[iref][p];

                float3[] Positions = GetPositionInAllTilts(ParticleCoords);
                for (int i = 0; i < Positions.Length; i++)
                    Positions[i] /= BinnedPixelSize;

                float3[] Angles = GetParticleAngleInAllTilts(ParticleCoords, particleAngles[iref][p]);

                Image ParticleCTFs = GetCTFsForOneParticle(options, ParticleCoords, CTFCoords, null);

                // Make projections

                float3[] ImageShifts = new float3[NTilts];

                for (int t = 0; t < NTilts; t++)
                {
                    ImageShifts[t] = new float3(Positions[t].X - (int)Positions[t].X, // +diff because we are shifting the projections into experimental data frame
                        Positions[t].Y - (int)Positions[t].Y,
                        Positions[t].Z - (int)Positions[t].Z);
                }

                Image ProjectionsFT = references[iref].Project(new int2(Size), Angles);

                ProjectionsFT.ShiftSlices(ImageShifts);
                ProjectionsFT.Multiply(ParticleCTFs);
                ParticleCTFs.Dispose();

                Image Projections = ProjectionsFT.AsIFFT().AndDisposeParent();
                Projections.RemapFromFT();


                // Insert projections into tilt series

                for (int t = 0; t < NTilts; t++)
                {
                    int2 IntPosition = new int2((int)Positions[t].X, (int)Positions[t].Y) - Size / 2;

                    float[] SimulatedData = SimulatedStack.GetHost(Intent.Write)[t];

                    float[] ImageData = Projections.GetHost(Intent.Read)[t];
                    for (int y = 0; y < Size; y++)
                    {
                        int PosY = y + IntPosition.Y;
                        if (PosY < 0 || PosY >= stackDimensions.Y)
                            continue;

                        for (int x = 0; x < Size; x++)
                        {
                            int PosX = x + IntPosition.X;
                            if (PosX < 0 || PosX >= stackDimensions.X)
                                continue;

                            SimulatedData[PosY * SimulatedStack.Dims.X + PosX] += ImageData[y * Size + x];
                        }
                    }
                }

                Projections.Dispose();
            }

            #endregion

            CTFCoords.Dispose();
        }

        return SimulatedStack;
    }
}