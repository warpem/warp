using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
//using Accord.Math;
using CommandLine;
using MathNet.Numerics.Random;
using Newtonsoft.Json.Linq;
using Warp;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;

namespace PCA3D
{
    class Program
    {
        static int DeviceID = 0;
        static int NThreads = 1;

        static string WorkingDirectory = @"";

        static int DimOri = -1;
        static int Dim = -1;
        static float AngPixOri = 1.699f;
        static float AngPix = 3.0f;

        static float Diameter = 300f;

        static float HighPass = 50;

        static string ParticlesStarPath = ""; 
        static string MaskPath = ""; 

        static string Symmetry = "C1";
        static Matrix3[] SymmetryMatrices;
        static int NSymmetry = 1;

        static int BatchLimit = 1024;

        static int NBatches = 0;
        static List<int> BatchSizes = new List<int>();
        static List<int[]> BatchOriginalRows = new List<int[]>();
        static List<int[]> BatchScaleGroups = new List<int[]>();
        static List<Image> BatchParticlesOri = new List<Image>();
        static List<Image> BatchParticlesMasked = new List<Image>();
        static List<Image> BatchCTFs = new List<Image>();
        static List<Image> BatchWeights = new List<Image>();
        static List<float[]> BatchScales = new List<float[]>();
        static List<Matrix3[]> BatchRotations = new List<Matrix3[]>();
        static List<int[]> BatchSubsets = new List<int[]>();
        static Image SpectralWeight;

        static int RetrainEveryNIters = 10;
        static int ProjectorPadding = 4;

        static Image DummySpectralWeights;

        static int NParticles = 0;
        static int NTilts = 1;
        static int NComponents = 8;
        static int NIterations = 60;

        static NoiseNet3DTorch Denoiser;

        static List<float[]>[] AllComponents = { new List<float[]>(), new List<float[]>() };

        static void Main(string[] args)
        {
            Options Options = new Options();

            #region Command line parsing

            {
                var Result = Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);

                if (Result.Tag == ParserResultType.NotParsed ||
                    Result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError ||
                                           e.Tag == ErrorType.HelpRequestedError))
                    return;

                WorkingDirectory = Environment.CurrentDirectory + "/";

                DeviceID = Options.DeviceID;
                NThreads = Options.NThreads;

                AngPixOri = Options.AngPixOri;
                AngPix = Options.AngPix;

                Diameter = Options.Diameter;

                ParticlesStarPath = Options.StarPath;
                MaskPath = Options.MaskPath;

                Symmetry = Options.Symmetry;

                BatchLimit = Options.BatchSize;

                NComponents = Options.NComponents;
                NIterations = Options.NIterations;
            }

            #endregion

            GPU.SetDevice(DeviceID);

            Denoiser = new NoiseNet3DTorch(new int3(64), new[] { DeviceID }, 4, 1, false, 128);

            Symmetry S = new Symmetry(Symmetry);
            SymmetryMatrices = S.GetRotationMatrices();
            NSymmetry = SymmetryMatrices.Length;

            #region Read STAR

            Console.WriteLine("Reading table...");

            Star TableIn = new Star(Path.Combine(WorkingDirectory, ParticlesStarPath));//.CreateSubset(Helper.ArrayOfSequence(0, 50000, 1));
            //TableIn.SortByKey(TableIn.GetColumn("rlnImageName").Select(s => s.Substring(s.IndexOf('@') + 1)).ToArray());
            //TableIn.RemoveRows(Helper.Combine(Helper.ArrayOfSequence(0, 10000, 1), Helper.ArrayOfSequence(20000, TableIn.RowCount, 1)));
            NParticles = 0;
            float3[] ParticleAngles = TableIn.GetRelionAngles().Select(a => a * Helper.ToRad).ToArray();
            float3[] ParticleShifts = TableIn.GetRelionOffsets();

            CTF[] ParticleCTFParams = TableIn.GetRelionCTF();
            {
                float MeanNorm = MathHelper.Mean(ParticleCTFParams.Select(p => (float)p.Scale));
                for (int p = 0; p < ParticleCTFParams.Length; p++)
                    ParticleCTFParams[p].Scale /= (decimal)MeanNorm;
            }

            int[] ParticleSubset = TableIn.GetColumn("rlnRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

            string[] ParticleNames = TableIn.GetColumn("rlnImageName");
            string[] UniqueMicrographs = Helper.GetUniqueElements(ParticleNames.Select(s => s.Substring(s.IndexOf('@') + 1))).ToArray();

            string[] ParticleGroupName = TableIn.HasColumn("rlnGroupName") ?
                                         TableIn.GetColumn("rlnGroupName") :
                                         Helper.ArrayOfFunction(i => i.ToString(), ParticleShifts.Length);

            NTilts = 1;// ParticleGroupName.Where(g => g == ParticleGroupName[0]).Count();
            BatchLimit = (BatchLimit / NTilts) * NTilts;

            Console.WriteLine("Done.\n");

            #endregion

            #region Prepare data

            Console.WriteLine("Loading and preparing data...");
            Console.Write("0/0");

            // Find out dimensions
            {
                int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[0]);
                string StackPath = Path.Combine(WorkingDirectory, ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1));

                MapHeader StackHeader = MapHeader.ReadFromFile(StackPath);

                DimOri = StackHeader.Dimensions.X;
                if (StackHeader.PixelSize.X != 1 && StackHeader.PixelSize.X != 0)
                {
                    AngPixOri = StackHeader.PixelSize.X;
                    Console.WriteLine($"Using pixel size from particle header, {AngPixOri:F4} A");
                }
                else
                {
                    Console.WriteLine($"Pixel size not found in particle header, using {AngPixOri:F4} A instead");
                }

                Dim = (int)Math.Round(DimOri * AngPixOri / AngPix / 2) * 2;
                AngPix = (float)DimOri / Dim * AngPixOri;   // Adjust pixel size to match rounded box size
            }

            {
                int NDone = 0;
                int NThreadsPreprocessing = 8;
                int PreprocessBatchLimit = (128 / NTilts) * NTilts;

                Image[] TBatchImages = Helper.ArrayOfFunction(i => new Image(new int3(DimOri, DimOri, PreprocessBatchLimit)), NThreadsPreprocessing);
                Image[] TBatchImagesFT = Helper.ArrayOfFunction(i => new Image(new int3(DimOri, DimOri, PreprocessBatchLimit), true, true), NThreadsPreprocessing);
                Image[] TBatchImagesFTCropped = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, PreprocessBatchLimit), true, true), NThreadsPreprocessing);
                Image[] TBatchAmpsReduced = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, 1), true), NThreadsPreprocessing);
                Image[] TBatchAmpsAll = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, 1), true), NThreadsPreprocessing);
                Image[] TBatchCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, PreprocessBatchLimit), true), NThreadsPreprocessing);
                Image[] TBatchWeights = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, PreprocessBatchLimit), true), NThreadsPreprocessing);
                Image[] TCTFCoords = Helper.ArrayOfFunction(i => CTF.GetCTFCoords(Dim, Dim), NThreadsPreprocessing);
                float[][][] TOriginalBuffer = Helper.ArrayOfFunction(i => new float[0][], NThreadsPreprocessing);
                int[] BatchPlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(DimOri, DimOri, 1), (uint)PreprocessBatchLimit), NThreadsPreprocessing);

                Helper.ForCPU(0, Math.Min(500, UniqueMicrographs.Length), NThreadsPreprocessing, threadID => GPU.SetDevice(DeviceID),
                    (imic, threadID) =>
                    {
                        int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                        string StackPath = Path.Combine(WorkingDirectory, ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1));

                        if (!File.Exists(StackPath))
                            throw new Exception($"No data found for {UniqueMicrographs[imic]}!");

                        int[] SliceIndices = Helper.IndexedSubset(ParticleNames, RowIndices).Select(s => int.Parse(s.Split(new[] { '@' })[0]) - 1).ToArray();
                        int NRelevant = SliceIndices.Length;

                        MapHeader OriginalHeader = MapHeader.ReadFromFile(StackPath);
                        if (TOriginalBuffer[threadID].Length < NRelevant)
                            TOriginalBuffer[threadID] = Helper.Combine(TOriginalBuffer[threadID], 
                                                                       Helper.ArrayOfFunction(i => new float[DimOri * DimOri],
                                                                                              NRelevant - TOriginalBuffer[threadID].Length));


                        float[][] RelevantStackData = IOHelper.ReadMapFloat(StackPath, new int2(1), 0, typeof(float), SliceIndices, null, TOriginalBuffer[threadID]);

                        float[][] BatchImagesData = TBatchImages[threadID].GetHost(Intent.Write);

                        float3[] MicShifts = Helper.IndexedSubset(ParticleShifts, RowIndices);

                        #region Rescale particles and pre-multiply them by B-factor, scale, and spectral whitening

                        Image RelevantFTCropped = new Image(IntPtr.Zero, new int3(Dim, Dim, NRelevant), true, true);
                        Image RelevantStackCTF = new Image(IntPtr.Zero, new int3(Dim, Dim, NRelevant), true);
                        Image RelevantStackWeights = new Image(IntPtr.Zero, new int3(Dim, Dim, NRelevant), true);

                        CTFStruct[] CTFParams = new CTFStruct[PreprocessBatchLimit];
                        CTFStruct[] CTFWeightParams = new CTFStruct[PreprocessBatchLimit];

                        for (int b = 0; b < NRelevant; b += PreprocessBatchLimit)
                        {
                            int CurBatch = Math.Min(PreprocessBatchLimit, NRelevant - b);

                            for (int i = 0; i < CurBatch; i++)
                                GPU.CopyHostToDevice(RelevantStackData[b + i],
                                                     TBatchImages[threadID].GetDeviceSlice(i, Intent.Write),
                                                     RelevantStackData[0].Length);

                            GPU.FFT(TBatchImages[threadID].GetDevice(Intent.Read),
                                    TBatchImagesFT[threadID].GetDevice(Intent.Write),
                                    new int3(DimOri, DimOri, 1),
                                    (uint)PreprocessBatchLimit,
                                    BatchPlanForw[threadID]);

                            GPU.CropFT(TBatchImagesFT[threadID].GetDevice(Intent.Read),
                                       TBatchImagesFTCropped[threadID].GetDevice(Intent.Write),
                                       new int3(DimOri, DimOri, 1),
                                       new int3(Dim, Dim, 1),
                                       (uint)CurBatch);

                            GPU.ShiftStackFT(TBatchImagesFTCropped[threadID].GetDevice(Intent.Read),
                                             TBatchImagesFTCropped[threadID].GetDevice(Intent.Write),
                                             new int3(Dim, Dim, 1),
                                             Helper.ToInterleaved(MicShifts.Skip(b).Take(CurBatch).Select(v => new float3(v.X * AngPixOri / AngPix + Dim / 2, 
                                                                                                                          v.Y * AngPixOri / AngPix + Dim / 2, 0)).ToArray()),
                                             (uint)CurBatch);

                            GPU.FourierBandpass(TBatchImagesFTCropped[threadID].GetDevice(Intent.Read),
                                                new int3(Dim, Dim, 1),
                                                AngPix * 2 / HighPass,
                                                1,
                                                AngPix * 2 / HighPass,
                                                (uint)CurBatch);

                            TBatchImagesFTCropped[threadID].Multiply(1f / (DimOri * DimOri));

                            for (int i = 0; i < CurBatch; i++)
                            {
                                int R = RowIndices[b + i];
                                ParticleCTFParams[R].PixelSize = (decimal)AngPix;

                                CTF CTFCopy = ParticleCTFParams[R].GetCopy();
                                CTFCopy.Bfactor = 0;
                                CTFCopy.BfactorDelta = 0;
                                CTFCopy.Scale = 1;
                                CTFParams[i] = CTFCopy.ToStruct();

                                CTF WeightCopy = ParticleCTFParams[R].GetCopy();
                                WeightCopy.Cs = 0;
                                WeightCopy.Defocus = 0;
                                WeightCopy.DefocusDelta = 0;
                                WeightCopy.Amplitude = 1;
                                CTFWeightParams[i] = WeightCopy.ToStruct();
                            }

                            // Unweighted CTFs match the model of the raw data
                            GPU.CreateCTF(TBatchCTF[threadID].GetDevice(Intent.Write),
                                          TCTFCoords[threadID].GetDevice(Intent.Read),
                                          IntPtr.Zero,
                                          (uint)TCTFCoords[threadID].ElementsComplex,
                                          CTFParams,
                                          false,
                                          (uint)CurBatch);

                            // Weights contain B-factor and scale
                            GPU.CreateCTF(TBatchWeights[threadID].GetDevice(Intent.Write),
                                          TCTFCoords[threadID].GetDevice(Intent.Read),
                                          IntPtr.Zero,
                                          (uint)TCTFCoords[threadID].ElementsComplex,
                                          CTFWeightParams,
                                          false,
                                          (uint)CurBatch);

                            // Images are pre-weighted, but not pre-CTF-multiplied
                            TBatchImagesFTCropped[threadID].Multiply(TBatchWeights[threadID]);

                            GPU.CopyDeviceToDevice(TBatchImagesFTCropped[threadID].GetDevice(Intent.Read),
                                                    RelevantFTCropped.GetDeviceSlice(b, Intent.Write),
                                                    RelevantFTCropped.ElementsSliceReal * CurBatch);

                            GPU.CopyDeviceToDevice(TBatchCTF[threadID].GetDevice(Intent.Read),
                                                    RelevantStackCTF.GetDeviceSlice(b, Intent.Write),
                                                    RelevantStackCTF.ElementsSliceReal * CurBatch);

                            GPU.CopyDeviceToDevice(TBatchWeights[threadID].GetDevice(Intent.Read),
                                                    RelevantStackWeights.GetDeviceSlice(b, Intent.Write),
                                                    RelevantStackWeights.ElementsSliceReal * CurBatch);
                        }

                        #endregion

                        lock (BatchSizes)
                        {
                            int NStored = 0;

                            while (NStored < NRelevant)
                            {
                                if (BatchSizes.Count == 0 || BatchSizes[BatchSizes.Count - 1] >= BatchLimit)
                                {
                                    BatchSizes.Add(0);
                                    BatchOriginalRows.Add(new int[BatchLimit]);
                                    BatchScaleGroups.Add(new int[BatchLimit]);
                                    BatchParticlesOri.Add(new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true, true));
                                    BatchCTFs.Add(new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true));
                                    BatchWeights.Add(new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true));
                                    BatchRotations.Add(Helper.ArrayOfFunction(i => new Matrix3(), BatchLimit));
                                    BatchScales.Add(new float[BatchLimit]);
                                    BatchSubsets.Add(new int[BatchLimit]);
                                }

                                int BatchID = BatchSizes.Count - 1;
                                int BatchEnd = BatchSizes[BatchID];
                                int SpaceLeft = BatchLimit - BatchEnd;

                                int StoringNow = Math.Min(SpaceLeft, NRelevant - NStored);

                                for (int s = 0; s < StoringNow; s++)
                                {
                                    BatchOriginalRows[BatchID][BatchEnd + s] = RowIndices[NStored + s];
                                    BatchScaleGroups[BatchID][BatchEnd + s] = imic * NTilts + ((NStored + s) % NTilts);
                                    BatchRotations[BatchID][BatchEnd + s] = Matrix3.Euler(ParticleAngles[RowIndices[NStored + s]]);
                                    BatchScales[BatchID][BatchEnd + s] = 1f;
                                    BatchSubsets[BatchID][BatchEnd + s] = ParticleSubset[RowIndices[NStored + s]];
                                }

                                GPU.CopyDeviceToDevice(RelevantFTCropped.GetDeviceSlice(NStored, Intent.Read),
                                                       BatchParticlesOri[BatchID].GetDeviceSlice(BatchEnd, Intent.Write),
                                                       RelevantFTCropped.ElementsSliceReal * StoringNow);
                                GPU.CopyDeviceToDevice(RelevantStackCTF.GetDeviceSlice(NStored, Intent.Read),
                                                       BatchCTFs[BatchID].GetDeviceSlice(BatchEnd, Intent.Write),
                                                       RelevantStackCTF.ElementsSliceReal * StoringNow);
                                GPU.CopyDeviceToDevice(RelevantStackWeights.GetDeviceSlice(NStored, Intent.Read),
                                                       BatchWeights[BatchID].GetDeviceSlice(BatchEnd, Intent.Write),
                                                       RelevantStackWeights.ElementsSliceReal * StoringNow);

                                BatchSizes[BatchID] += StoringNow;
                                NStored += StoringNow;
                            }

                            NParticles += RowIndices.Length;

                            ClearCurrentConsoleLine();
                            Console.Write($"{++NDone}/{UniqueMicrographs.Length}, {GPU.GetFreeMemory(DeviceID)} MB");
                        }

                        RelevantFTCropped.Dispose();
                        RelevantStackCTF.Dispose();
                        RelevantStackWeights.Dispose();
                    }, null);

                for (int i = 0; i < NThreadsPreprocessing; i++)
                {
                    TBatchImages[i].Dispose();
                    TBatchImagesFT[i].Dispose();
                    TBatchImagesFTCropped[i].Dispose();
                    TBatchAmpsReduced[i].Dispose();
                    TBatchAmpsAll[i].Dispose();
                    TBatchCTF[i].Dispose();
                    TBatchWeights[i].Dispose();
                    TCTFCoords[i].Dispose();
                    GPU.DestroyFFTPlan(BatchPlanForw[i]);
                }
            }
            Console.WriteLine("");

            NBatches = BatchSizes.Count;

            Console.WriteLine("Done.\n");

            #endregion

            #region Mask

            Console.WriteLine("Loading mask...");

            Image Mask = Image.FromFile(Path.Combine(WorkingDirectory, MaskPath));
            Image MaskHard = null;
            int[] MaskIndices, MaskIndicesHard;
            {
                if (Mask.Dims.X != DimOri)
                {
                    Image MaskPadded = Mask.AsPadded(new int3(DimOri));
                    Mask.Dispose();
                    Mask = MaskPadded;
                }

                Image MaskScaled = Mask.AsScaled(new int3(Dim)).AndDisposeParent();
                MaskScaled.Max(0);
                MaskScaled.Min(1);

                Mask = MaskScaled;
                MaskHard = Mask.GetCopy();
                //MaskScaled.Binarize(0.5f);

                //MaskHard = FSC.MakeSoftMask(MaskScaled, 2, 0);
                //MaskScaled.Dispose();

                //Mask = FSC.MakeSoftMask(MaskHard, 0, 3);
                Mask.WriteMRC("d_mask.mrc", true);
            }

            {
                float[] MaskData = Mask.GetHostContinuousCopy();
                List<int> MaskIndicesList = new List<int>();
                for (int i = 0; i < MaskData.Length; i++)
                    if (MaskData[i] > 0)
                        MaskIndicesList.Add(i);
                MaskIndices = MaskIndicesList.ToArray();
            }

            {
                float[] MaskData = MaskHard.GetHostContinuousCopy();
                List<int> MaskIndicesList = new List<int>();
                for (int i = 0; i < MaskData.Length; i++)
                    if (MaskData[i] > 0)
                        MaskIndicesList.Add(i);
                MaskIndicesHard = MaskIndicesList.ToArray();
            }

            Console.WriteLine("Done.\n");

            Dim = Mask.Dims.X;

            #endregion

            Console.WriteLine(GPU.GetFreeMemory(0));

            #region Reconstruct and subtract average

            {
                Console.WriteLine("Reconstructing average and estimating per-particle scale...");

                Image[] VolumeAverage = new Image[2];

                for (int irepeat = 0; irepeat < 4; irepeat++)
                {
                    // Make reconstructions with current best guesses for scaling factors
                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        Projector Reconstructor = new Projector(new int3(Dim), ProjectorPadding);
                        Image Particles = new Image(IntPtr.Zero, BatchParticlesOri[0].Dims, true, true);
                        Image CTFs = new Image(IntPtr.Zero, BatchCTFs[0].Dims, true);

                        Helper.ForCPU(0, NBatches, 1, threadID => GPU.SetDevice(DeviceID),
                            (batchID, threadID) =>
                            {
                                GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                       Particles.GetDevice(Intent.Write),
                                                       Particles.ElementsReal);
                                GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                       CTFs.GetDevice(Intent.Write),
                                                       CTFs.ElementsReal);

                                CTFs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());
                                CTFs.Multiply(BatchScales[batchID]);

                                // Particles are pre-weighted, so only need CTF mul
                                Particles.Multiply(CTFs);

                                // Weights are CTF^2 * Bfac
                                CTFs.Multiply(CTFs);
                                CTFs.Multiply(BatchWeights[batchID]);

                                //if (SpectralWeight != null)
                                //{
                                //    Particles.MultiplySlices(SpectralWeight);
                                //    CTFs.MultiplySlices(SpectralWeight);
                                //}

                                foreach (var m in SymmetryMatrices)
                                    Reconstructor.BackProject(Particles,
                                                              CTFs,
                                                              BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                              Matrix2.Identity());

                                //BatchParticlesOri[batchID].FreeDevice();
                                //BatchCTFs[batchID].FreeDevice();
                                //BatchWeights[batchID].FreeDevice();
                            }, null);

                        Particles.Dispose();
                        CTFs.Dispose();

                        VolumeAverage[ihalf] = Reconstructor.Reconstruct(false, "C1");
                        Reconstructor.Dispose();

                        VolumeAverage[ihalf].MaskSpherically(Dim - 16, 8, true);
                        VolumeAverage[ihalf].WriteMRC(Path.Combine(WorkingDirectory, $"pc_{0:D2}_{ihalf + 1}.mrc"), true);
                    }

                    if (true)
                    {
                        Image Denoised1, Denoised2, DenoisedCombined;
                        TrainDenoiser(Denoiser, VolumeAverage[0], VolumeAverage[1], irepeat == 0, irepeat == 0 ? 1000 : 0, out Denoised1, out Denoised2, out DenoisedCombined);

                        VolumeAverage[0].Dispose();
                        VolumeAverage[1].Dispose();

                        VolumeAverage[0] = DenoisedCombined.GetCopyGPU();
                        VolumeAverage[1] = DenoisedCombined.GetCopyGPU();

                        VolumeAverage[0].WriteMRC(Path.Combine(WorkingDirectory, $"denoised_pc_{0:D2}_1.mrc"), true);
                        VolumeAverage[1].WriteMRC(Path.Combine(WorkingDirectory, $"denoised_pc_{0:D2}_2.mrc"), true);                        

                        Denoised1.Dispose();
                        Denoised2.Dispose();
                        DenoisedCombined.Dispose();
                    }

                    Image NextSpectralWeight = new Image(new int3(Dim, Dim, 1), true);

                    // Calculate scaling factors
                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        Image VolumeMasked = VolumeAverage[0].GetCopyGPU();
                        VolumeMasked.Multiply(Mask);
                        Projector Projector = new Projector(VolumeMasked, ProjectorPadding, true);
                        VolumeMasked.Dispose();
                        //Projector.PutTexturesOnDevice();

                        Image Particles = new Image(IntPtr.Zero, BatchParticlesOri[0].Dims, true, true);
                        Image CTFs = new Image(IntPtr.Zero, BatchCTFs[0].Dims, true);
                        Image Refs = new Image(IntPtr.Zero, Particles.Dims, true, true);
                        Image ReducedAmps = new Image(new int3(Dim, Dim, 1), true);

                        for (int batchID = 0; batchID < NBatches; batchID++)
                        {
                            GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                    Particles.GetDevice(Intent.Write),
                                                    Particles.ElementsReal);
                            GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                    CTFs.GetDevice(Intent.Write),
                                                    CTFs.ElementsReal);

                            // Particles are pre-weighted
                            // CTFs are CTF * Bfac
                            CTFs.Multiply(BatchWeights[batchID]);

                            float3[] Angles = BatchRotations[batchID].Select(a => Matrix3.EulerFromMatrix(a)).ToArray();
                            float[] Result = new float[BatchLimit];

                            GPU.PCALeastSq(Result,
                                            Particles.GetDevice(Intent.Read),
                                            CTFs.GetDevice(Intent.Read),
                                            SpectralWeight == null ? IntPtr.Zero : SpectralWeight.GetDevice(Intent.Read),
                                            Dim,
                                            Dim / 2,
                                            new float[BatchSizes[batchID] * 2],
                                            Helper.ToInterleaved(Angles),
                                            Matrix2.Scale(Projector.Oversampling, Projector.Oversampling).ToVec(),
                                            Projector.t_DataRe,
                                            Projector.t_DataIm,
                                            Projector.DimsOversampled.X,
                                            BatchSizes[batchID],
                                            NTilts);

                            //BatchParticlesOri[batchID].FreeDevice();
                            //BatchCTFs[batchID].FreeDevice();
                            //BatchWeights[batchID].FreeDevice();

                            for (int i = 0; i < BatchSizes[batchID]; i++)
                            {
                                if (BatchSubsets[batchID][i] == ihalf)
                                    BatchScales[batchID][i] = (Result[i] > 0.5f && Result[i] < 1.5f) ? Result[i] : 0f;
                            }

                            Projector.Project(new int2(Dim), Angles, Refs);
                            Refs.Multiply(CTFs);
                            Refs.Multiply(Result);
                            Refs.Subtract(Particles);
                            //Refs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());

                            GPU.Amplitudes(Refs.GetDevice(Intent.Read), CTFs.GetDevice(Intent.Write), Refs.ElementsComplex);
                            GPU.ReduceAdd(CTFs.GetDevice(Intent.Read),
                                          ReducedAmps.GetDevice(Intent.Write),
                                          (uint)CTFs.ElementsSliceReal,
                                          (uint)BatchSizes[batchID],
                                          1);
                            NextSpectralWeight.Add(ReducedAmps);
                        }

                        Projector.Dispose();
                        Particles.Dispose();
                        CTFs.Dispose();
                        Refs.Dispose();
                        ReducedAmps.Dispose();
                    }

                    float MedianScale = MathHelper.Mean(Helper.Combine(BatchScales).Where(v => v != 0f));
                    Console.WriteLine(MedianScale);
                    for (int i = 0; i < BatchScales.Count; i++)
                        BatchScales[i] = BatchScales[i].Select(v => v / MedianScale).ToArray();

                    // Calculate spectral weights from rotational average of (particle - ref * CTF)
                    {
                        float[] Amps1D = new float[Dim / 2];
                        float[] Samples1D = new float[Dim / 2];
                        float[] Amps2D = NextSpectralWeight.GetHost(Intent.Read)[0];

                        Helper.ForEachElementFT(new int2(Dim), (x, y, xx, yy, r, angle) =>
                        {
                            int idx = (int)Math.Round(r);
                            if (idx < Dim / 2)
                            {
                                float W1 = r - (int)r;
                                float W0 = 1 - W1;
                                Amps1D[idx] += Amps2D[y * (Dim / 2 + 1) + x] * W0;
                                Samples1D[idx] += W0;
                                Amps1D[Math.Min(Amps1D.Length - 1, idx + 1)] += Amps2D[y * (Dim / 2 + 1) + x] * W1;
                                Samples1D[Math.Min(Amps1D.Length - 1, idx + 1)] += W1;
                            }
                        });

                        for (int i = 0; i < Amps1D.Length; i++)
                            Amps1D[i] = 1;// / (Amps1D[i] / Samples1D[i]);
                        Amps1D[0] = 0f;

                        float MaxAmps = MathHelper.Max(Amps1D);
                        for (int i = 0; i < Amps1D.Length; i++)
                            Amps1D[i] /= MaxAmps;

                        Helper.ForEachElementFT(new int2(Dim), (x, y, xx, yy, r, angle) =>
                        {
                            Amps2D[y * (Dim / 2 + 1) + x] = MathHelper.Lerp(Amps1D[Math.Min((int)r, Dim / 2 - 1)],
                                                                            Amps1D[Math.Min((int)r + 1, Dim / 2 - 1)],
                                                                            r - (int)r);
                        });

                        SpectralWeight?.Dispose();
                        SpectralWeight = NextSpectralWeight;
                        SpectralWeight.WriteMRC($"d_specweight_{irepeat:D2}.mrc", true);
                    }
                }

                {
                    Image Norm1 = VolumeAverage[0].GetCopy();
                    Norm1.Multiply(Mask);
                    float L2 = MathHelper.L2(Norm1.GetHostContinuousCopy());
                    Norm1.Multiply(1f / L2);
                    AllComponents[0].Add(Norm1.GetHostContinuousCopy());
                    Norm1.Dispose();

                    Image Norm2 = VolumeAverage[1].GetCopy();
                    Norm2.Multiply(Mask);
                    L2 = MathHelper.L2(Norm2.GetHostContinuousCopy());
                    Norm2.Multiply(1f / L2);
                    AllComponents[1].Add(Norm2.GetHostContinuousCopy());
                    Norm2.Dispose();
                }

                // Subtract average projections from raw particles
                // While we're at it, also make reconstructions from subtracted, and subtracted & masked particles
                for (int ihalf = 0; ihalf < 2; ihalf++)
                {
                    Image VolumeMasked = VolumeAverage[ihalf].GetCopyGPU();
                    //VolumeMasked.Multiply(Mask);
                    Projector ProjectorAverage = new Projector(VolumeMasked, ProjectorPadding, true);
                    VolumeMasked.Dispose();

                    Projector ReconstructorSubtracted = new Projector(new int3(Dim), ProjectorPadding);

                    Image Refs = new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true, true);
                    Image CTFs = new Image(IntPtr.Zero, new int3(Dim, Dim, BatchLimit), true);

                    for (int batchID = 0; batchID < NBatches; batchID++)
                    {
                        // Subtract average projections and make masked copies
                        {
                            ProjectorAverage.Project(new int2(Dim), BatchRotations[batchID].Select(a => Matrix3.EulerFromMatrix(a)).ToArray(), Refs);
                            Refs.Multiply(BatchCTFs[batchID]);
                            Refs.Multiply(BatchWeights[batchID]);
                            Refs.Multiply(BatchScales[batchID]);
                            Refs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());

                            BatchParticlesOri[batchID].Subtract(Refs);

                            GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                   Refs.GetDevice(Intent.Write),
                                                   Refs.ElementsReal);
                            GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                   CTFs.GetDevice(Intent.Write),
                                                   CTFs.ElementsReal);

                            CTFs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());
                            CTFs.Multiply(BatchScales[batchID]);

                            // Particles are pre-weighted, so only need CTF mul
                            Refs.Multiply(CTFs);

                            // Weights are CTF^2 * Bfac
                            CTFs.Multiply(CTFs);
                            CTFs.Multiply(BatchWeights[batchID]);

                            foreach (var m in SymmetryMatrices)
                                ReconstructorSubtracted.BackProject(Refs,
                                                                    CTFs,
                                                                    BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                                    Matrix2.Identity());

                            //BatchParticlesOri[batchID].FreeDevice();
                            //BatchParticlesMasked[batchID].FreeDevice();
                            //BatchCTFs[batchID].FreeDevice();
                        }
                    }

                    Refs.Dispose();
                    CTFs.Dispose();
                    ProjectorAverage.Dispose();

                    Image AverageSubtracted = ReconstructorSubtracted.Reconstruct(false, "C1");
                    ReconstructorSubtracted.Dispose();
                    AverageSubtracted.WriteMRC(Path.Combine(WorkingDirectory, $"subtracted_half{ihalf + 1}.mrc"), true);
                    AverageSubtracted.Dispose();
                }

                Console.WriteLine("Done.\n");
            }

            #endregion

            Star SettingsTable = new Star(new[] { "wrpParticleTable", "wrpSymmetry", "wrpPixelSize", "wrpDiameter", "wrpMask" });
            SettingsTable.AddRow(new string[] { ParticlesStarPath, Symmetry, AngPixOri.ToString(), Diameter.ToString(), MaskPath });

            Star WeightsTable = new Star(new string[0]);
            WeightsTable.AddColumn($"wrpPCA{(0):D2}", Helper.ArrayOfConstant("1.000", NParticles * NSymmetry));

            int[] Hist = MathHelper.Histogram(Helper.Combine(BatchScales.ToArray()), 100, 0, 2);

            
            RandomNormal RandN = new RandomNormal(123);

            List<float[]> AllNormalizedScores = new List<float[]>();

            int PlanForwRec, PlanBackRec, PlanForwCTF;
            Projector.GetPlans(new int3(Dim), ProjectorPadding, out PlanForwRec, out PlanBackRec, out PlanForwCTF);


            for (int icomponent = 0; icomponent < NComponents; icomponent++)
            {
                Console.WriteLine($"PC {icomponent}:");

                Image Particles = new Image(IntPtr.Zero, BatchParticlesOri[0].Dims, true, true);
                Image CTFs = new Image(IntPtr.Zero, BatchCTFs[0].Dims, true);

                float[] PC = Helper.ArrayOfFunction(i => RandN.NextSingle(0, 1), (int)new int3(Dim).Elements());
                Image[] CurVolumes = { new Image(PC, new int3(Dim)), new Image(PC, new int3(Dim)) };
                CurVolumes[0].Bandpass(0, 0, true, 0.05f);
                CurVolumes[1].Bandpass(0, 0, true, 0.05f);

                float RMax = MathF.Max(0.05f * Dim / 2, AngPix / HighPass * Dim);

                float[] AllScores = new float[NParticles * NSymmetry];

                Console.Write($"0/{NIterations}");

                for (int iiter = 0; iiter < NIterations; iiter++)
                {
                    Image[] NextVolumes = new Image[2];

                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        Image Volume = CurVolumes[ihalf];
                        if (iiter > RetrainEveryNIters)
                            Volume.Multiply(Mask);
                        if (iiter % RetrainEveryNIters == 0)
                            Volume.WriteMRC($"d_pc{icomponent:D2}_it{iiter:D3}_{ihalf}.mrc", true);

                        Projector VolumeProjector = new Projector(Volume, ProjectorPadding, true);
                        //VolumeProjector.PutTexturesOnDevice();

                        Projector NextReconstructor = new Projector(new int3(Dim), ProjectorPadding);

                        for (int batchID = 0; batchID < NBatches; batchID++)
                        {
                            int CurBatch = BatchSizes[batchID];
                            float[] Scores = new float[BatchLimit / NTilts];

                            GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                    Particles.GetDevice(Intent.Write),
                                                    Particles.ElementsReal);
                            GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                    CTFs.GetDevice(Intent.Write),
                                                    CTFs.ElementsReal);

                            // Particles are pre-weighted
                            // CTFs are CTF * Bfac
                            CTFs.Multiply(BatchWeights[batchID]);
                            //CTFs.Multiply(BatchScales[batchID]);

                            for (int im = 0; im < NSymmetry; im++)
                            {
                                Matrix3 MSym = SymmetryMatrices[im];
                                float3[] Angles = BatchRotations[batchID].Take(CurBatch).Select(a => Matrix3.EulerFromMatrix(a * MSym)).ToArray();

                                GPU.PCALeastSq(Scores,
                                               Particles.GetDevice(Intent.Read),
                                               CTFs.GetDevice(Intent.Read),
                                               SpectralWeight.GetDevice(Intent.Read),
                                               Dim,
                                               RMax,
                                               new float[CurBatch * 2],
                                               Helper.ToInterleaved(Angles),
                                               Matrix2.Scale(VolumeProjector.Oversampling, VolumeProjector.Oversampling).ToVec(),
                                               VolumeProjector.t_DataRe,
                                               VolumeProjector.t_DataIm,
                                               VolumeProjector.DimsOversampled.X,
                                               CurBatch / NTilts,
                                               NTilts);

                                for (int i = 0; i < BatchLimit; i++)
                                    if (BatchScales[batchID][i] == 0)
                                        Scores[i] = 0;

                                for (int p = 0; p < CurBatch / NTilts; p++)
                                    for (int t = 0; t < NTilts; t++)
                                        if (BatchSubsets[batchID][p * NTilts + t] == ihalf)
                                            AllScores[(batchID * BatchLimit + p * NTilts + t) * NSymmetry + im] = Scores[p];

                                GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                       Particles.GetDevice(Intent.Write),
                                                       Particles.ElementsReal);
                                GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                       CTFs.GetDevice(Intent.Write),
                                                       CTFs.ElementsReal);

                                CTFs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());
                                //CTFs.Multiply(BatchScales[batchID]);
                                CTFs.Multiply(Scores);

                                // Particles are pre-weighted, so only need CTF mul
                                Particles.Multiply(CTFs);

                                // Weights are CTF^2 * Bfac
                                CTFs.Multiply(CTFs);
                                CTFs.Multiply(BatchWeights[batchID]);

                                //if (SpectralWeight != null)
                                //{
                                //    Particles.MultiplySlices(SpectralWeight);
                                //    CTFs.MultiplySlices(SpectralWeight);
                                //}

                                foreach (var m in SymmetryMatrices)
                                    NextReconstructor.BackProject(Particles,
                                                                  CTFs,
                                                                  Angles,
                                                                  Matrix2.Identity(),
                                                                  0,
                                                                  (int)RMax + 16);
                            }
                        }

                        Image NextVolume = NextReconstructor.Reconstruct(false, "C1", PlanForwRec, PlanBackRec, PlanForwCTF);
                        NextVolume.MaskSpherically(Dim - 16, 8, true);
                        if (iiter % RetrainEveryNIters == 0)
                            NextVolume.WriteMRC($"d_nextvolume{icomponent:D2}_it{iiter:D3}_{ihalf}.mrc", true);

                        VolumeProjector.Dispose();
                        NextReconstructor.Dispose();

                        NextVolumes[ihalf] = NextVolume;
                    }

                    if (iiter >= NIterations - 1)
                    {
                        NextVolumes[0].Dispose();
                        NextVolumes[1].Dispose();
                    }
                    else
                    {
                        {
                            Image MaskedHalf1 = NextVolumes[0].GetCopyGPU();
                            Image MaskedHalf2 = NextVolumes[1].GetCopyGPU();

                            if (iiter >= RetrainEveryNIters)
                            {
                                MaskedHalf1.Multiply(Mask);
                                MaskedHalf2.Multiply(Mask);
                            }

                            float[] MaskedFSC = FSC.GetFSC(MaskedHalf1, MaskedHalf2);
                            MaskedHalf1.Dispose();
                            MaskedHalf2.Dispose();

                            float GlobalShell = MathF.Max(FSC.GetCutoffShell(MaskedFSC, 0.05f), AngPix / HighPass * Dim);
                            NextVolumes[0].Bandpass(0, GlobalShell / (Dim / 2), true, 0.05f);
                            NextVolumes[1].Bandpass(0, GlobalShell / (Dim / 2), true, 0.05f);

                            RMax = GlobalShell + 0.05f * Dim / 2;
                        }

                        {
                            bool RetrainDenoiser = (iiter % RetrainEveryNIters == 0) || iiter == NIterations - 2;

                            if (RetrainDenoiser)
                                Console.Write("\n");

                            Image[] NextDenoised = new Image[2];
                            Image NextDenoisedCombined;
                            TrainDenoiser(Denoiser, NextVolumes[0], NextVolumes[1], false, RetrainDenoiser ? (iiter == 0 ? 1000 : 200) : 0, out NextDenoised[0], out NextDenoised[1], out NextDenoisedCombined);

                            if (iiter > 5)// NIterations - DenoiseEveryNIters)
                                for (int ihalf = 0; ihalf < 2; ihalf++)
                                {
                                    NextDenoised[ihalf].Dispose();
                                    NextDenoised[ihalf] = NextDenoisedCombined.GetCopyGPU();
                                }

                            Image NextVolumeAvg1, NextVolumeAvg2;
                            //FSC.AverageLowFrequencies(NextDenoised[0], NextDenoised[1], (int)(AngPix / 60 * Dim), out NextVolumeAvg1, out NextVolumeAvg2);
                            FSC.AverageLowFrequencies(NextDenoised[0], NextDenoised[1], Dim / 2, out NextVolumeAvg1, out NextVolumeAvg2);

                            for (int ihalf = 0; ihalf < 2; ihalf++)
                            {
                                NextDenoised[ihalf].Dispose();
                                NextVolumes[ihalf].Dispose();
                                CurVolumes[ihalf].Dispose();
                                CurVolumes[ihalf] = ihalf == 0 ? NextVolumeAvg1 : NextVolumeAvg2;
                            }

                            NextDenoisedCombined.Dispose();

                            //CurVolumes[0].WriteMRC($"d_denoisedvolume{icomponent:D2}_it{iiter:D3}_0.mrc", true);
                            //CurVolumes[1].WriteMRC($"d_denoisedvolume{icomponent:D2}_it{iiter:D3}_1.mrc", true);
                        }

                        {
                            Image NextAveraged = CurVolumes[0].GetCopy();
                            NextAveraged.Add(CurVolumes[1]);
                            NextAveraged.Multiply(0.5f);
                            NextAveraged.Multiply(Mask);

                            float L2 = MathHelper.L2(NextAveraged.GetHostContinuousCopy());
                            NextAveraged.Dispose();

                            CurVolumes[0].Multiply(1f / L2);
                            CurVolumes[1].Multiply(1f / L2);
                        }
                    }

                    float Orthogonality = 0;
                    for (int icomp = 0; icomp < AllComponents[0].Count; icomp++)
                    {
                        for (int ihalf = 0; ihalf < 2; ihalf++)
                        {
                            float DotP = MathHelper.DotProduct(CurVolumes[ihalf].GetHostContinuousCopy(), AllComponents[ihalf][icomp]);

                            //CurVolumes[ihalf].TransformValues((i, v) => v - DotP * AllComponents[ihalf][icomp][i]);

                            Orthogonality += MathF.Abs(DotP);
                        }
                    }

                    Orthogonality /= AllComponents[0].Count * 2;
                    Orthogonality = 1 - Math.Abs(Orthogonality);

                    ClearCurrentConsoleLine();
                    Console.Write($"{iiter + 1}/{NIterations}, orthogonality = {Orthogonality:F7}");//, convergence = {DotWithPreviousIteration:F7}");
                }
                Console.Write("\n");

                // Subtract PC projections from particles
                if (true)
                {
                    // Subtract average projections from raw particles, and make a real space-masked copy for comparisons later
                    // While we're at it, also make reconstructions from subtracted, and subtracted & masked particles
                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        Projector ProjectorAverage = new Projector(CurVolumes[0], ProjectorPadding, true);
                        Projector ReconstructorSubtracted = new Projector(new int3(Dim), ProjectorPadding);

                        for (int batchID = 0; batchID < NBatches; batchID++)
                        {
                            // Subtract average projections and make masked copies
                            {
                                ProjectorAverage.Project(new int2(Dim), BatchRotations[batchID].Select(a => Matrix3.EulerFromMatrix(a)).ToArray(), Particles);
                                Particles.Multiply(BatchCTFs[batchID]);
                                Particles.Multiply(BatchWeights[batchID]);
                                //Particles.Multiply(BatchScales[batchID]);
                                Particles.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());

                                float[] Scores = new float[BatchLimit];
                                for (int p = 0; p < BatchSizes[batchID]; p++)
                                    Scores[p] = AllScores[(batchID * BatchLimit + p) * NSymmetry + 0];
                                Particles.Multiply(Scores);

                                BatchParticlesOri[batchID].Subtract(Particles);

                                GPU.CopyDeviceToDevice(BatchParticlesOri[batchID].GetDevice(Intent.Read),
                                                       Particles.GetDevice(Intent.Write),
                                                       Particles.ElementsReal);
                                GPU.CopyDeviceToDevice(BatchCTFs[batchID].GetDevice(Intent.Read),
                                                       CTFs.GetDevice(Intent.Write),
                                                       CTFs.ElementsReal);

                                CTFs.Multiply(BatchSubsets[batchID].Select(v => (v == ihalf) ? 1f : 0f).ToArray());
                                CTFs.Multiply(BatchScales[batchID]);

                                // Particles are pre-weighted, so only need CTF mul
                                Particles.Multiply(CTFs);

                                // Weights are CTF^2 * Bfac
                                CTFs.Multiply(CTFs);
                                CTFs.Multiply(BatchWeights[batchID]);

                                foreach (var m in SymmetryMatrices)
                                    ReconstructorSubtracted.BackProject(Particles,
                                                                        CTFs,
                                                                        BatchRotations[batchID].Take(BatchSizes[batchID]).Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray(),
                                                                        Matrix2.Identity());

                                //BatchParticlesOri[batchID].FreeDevice();
                                //BatchParticlesMasked[batchID].FreeDevice();
                                //BatchCTFs[batchID].FreeDevice();
                            }
                        }

                        Particles.Dispose();
                        CTFs.Dispose();
                        ProjectorAverage.Dispose();

                        Image AverageSubtracted = ReconstructorSubtracted.Reconstruct(false, "C1");
                        ReconstructorSubtracted.Dispose();
                        AverageSubtracted.WriteMRC(Path.Combine(WorkingDirectory, $"subtracted_pc{icomponent:D2}_half{ihalf + 1}.mrc"), true);
                        AverageSubtracted.Dispose();
                    }
                }


                for (int ihalf = 0; ihalf < 2; ihalf++)
                {
                    AllComponents[ihalf].Add(CurVolumes[ihalf].GetHostContinuousCopy());

                    CurVolumes[ihalf].WriteMRC(Path.Combine(WorkingDirectory, $"denoised_pc_{(icomponent + 1):D2}_{ihalf + 1}.mrc"), true);
                    CurVolumes[ihalf].Dispose();
                }

                Particles.Dispose();
                CTFs.Dispose();

                AllNormalizedScores.Add(AllScores);

                float[] InterpolatedScores = new float[NParticles * NSymmetry];

                //{
                //    MathHelper.NormalizeL2InPlace(PC);
                //    AllComponents.Add(PC);

                //    //MathHelper.NormalizeL2InPlace(PCHard);
                //    //AllComponentsHard.Add(PCHard);

                //    for (int i = 0; i < MaskIndices.Length; i++)
                //        VolumeData[MaskIndices[i]] = PC[i];

                //    Image Volume = new Image(VolumeData, new int3(Dim));
                //    Volume.Bandpass(0, 1, true);
                //    Image VolumeFT = Volume.AsFFT(true);
                //    Volume.Dispose();
                //    float VolumeL2 = (float)Math.Sqrt(VolumeFT.GetHostContinuousCopy().Select(v => v * v).Sum() * 2);
                //    VolumeFT.Multiply(1 / VolumeL2 / Dim);
                //    Volume = VolumeFT.AsIFFT(true);
                //    VolumeFT.Dispose();

                //    Volume.WriteMRC(Path.Combine(WorkingDirectory, $"pc_{(icomponent + 1):D2}.mrc"), true);
                //    Volume.Dispose();

                //    float NormFactor = 1f / Dim;// /= Dim * Dim;


                //    {
                //        float[] OrderedWeights = new float[NParticles * NSymmetry];
                //        for (int p = 0; p < NParticles; p++)
                //        {
                //            int b = p / BatchLimit;
                //            int bp = p % BatchLimit;
                //            int r = BatchOriginalRows[b][bp];

                //            for (int s = 0; s < NSymmetry; s++)
                //                OrderedWeights[r * NSymmetry + s] = AllScores[p * NSymmetry + s];
                //        }

                //        WeightsTable.AddColumn($"wrpPCA{(icomponent + 1):D2}", OrderedWeights.Select(v => (v / NormFactor).ToString(CultureInfo.InvariantCulture)).ToArray());
                //    }

                //    Star.SaveMultitable(Path.Combine(WorkingDirectory, "3dpca.star"), new Dictionary<string, Star>
                //    {
                //        { "settings", SettingsTable},
                //        { "weights", WeightsTable}
                //    });

                //    Console.WriteLine(GPU.GetFreeMemory(DeviceID) + " MB");
                //}
            }


            //Console.Read();
        }

        public static void TrainDenoiser(NoiseNet3DTorch model, Image half1, Image half2, bool fromScratch, int niterations, out Image denoised1, out Image denoised2, out Image denoisedCombined)
        {
            Image MapHalf1 = half1;
            Image MapHalf2 = half2;

            #region Prepare data

            int3 BoundingBox = MapHalf1.Dims / 2;
            float2 MeanStdForDenoising;

            // Normalize to mean = 0, std = 1 taking the central 1/8 of the map as reference
            {
                Image Map1Center = MapHalf1.AsPadded(MapHalf1.Dims / 2);
                Image Map2Center = MapHalf2.AsPadded(MapHalf2.Dims / 2);

                MeanStdForDenoising = MathHelper.MeanAndStd(Helper.Combine(Map1Center.GetHostContinuousCopy(), Map2Center.GetHostContinuousCopy()));

                Map1Center.Dispose();
                Map2Center.Dispose();
            }

            MapHalf1.Add(-MeanStdForDenoising.X);
            MapHalf1.Multiply(1f / MeanStdForDenoising.Y);
            MapHalf2.Add(-MeanStdForDenoising.X);
            MapHalf2.Multiply(1f / MeanStdForDenoising.Y);

            Image MapHalf1Ori = MapHalf1.GetCopy();
            Image MapHalf2Ori = MapHalf2.GetCopy();

            GPU.PrefilterForCubic(MapHalf1.GetDevice(Intent.ReadWrite), MapHalf1.Dims);
            GPU.PrefilterForCubic(MapHalf2.GetDevice(Intent.ReadWrite), MapHalf2.Dims);

            ulong[] Texture1 = new ulong[1], TextureArray1 = new ulong[1];
            GPU.CreateTexture3D(MapHalf1.GetDevice(Intent.Read), MapHalf1.Dims, Texture1, TextureArray1, true);
            MapHalf1.FreeDevice();

            ulong[] Texture2 = new ulong[1], TextureArray2 = new ulong[1];
            GPU.CreateTexture3D(MapHalf2.GetDevice(Intent.Read), MapHalf2.Dims, Texture2, TextureArray2, true);
            MapHalf2.FreeDevice();

            #endregion

            #region Training

            int GPUNetwork = model.Devices[0];

            int3 Dim = model.BoxDimensions;
            int BatchSize = model.BatchSize;
            int NIterations = niterations;

            float LearningRateStart = fromScratch ? 1e-4f : 1e-4f;
            float LearningRateFinish = fromScratch ? 1e-5f : 1e-5f;

            //Console.WriteLine($"Will train for {NIterations} iterations");

            Random Rand = new Random((int)(MeanStdForDenoising.X * 1e8));

            Image ExtractedSource = new Image(new int3(Dim.X, Dim.Y, Dim.Z * BatchSize));
            Image ExtractedTarget = new Image(new int3(Dim.X, Dim.Y, Dim.Z * BatchSize));

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Queue<float> Losses = new Queue<float>();

            Image PredictedData = null;
            float[] Loss = null;

            for (int iter = 0; iter < NIterations; iter++)
            {
                {
                    int3 DimsMap = MapHalf1.Dims;
                    int3 Margin = (DimsMap - BoundingBox) / 2;

                    float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), BatchSize);

                    float3[] Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                            (float)Rand.NextDouble() * 360,
                                                                            (float)Rand.NextDouble() * 360) * Helper.ToRad, BatchSize);

                    GPU.Rotate3DExtractAt(Texture1[0],
                                            MapHalf1.Dims,
                                            ExtractedSource.GetDevice(Intent.Write),
                                            Dim,
                                            Helper.ToInterleaved(Angle),
                                            Helper.ToInterleaved(Position),
                                            (uint)BatchSize);
                    GPU.Rotate3DExtractAt(Texture2[0],
                                            MapHalf2.Dims,
                                            ExtractedTarget.GetDevice(Intent.Write),
                                            Dim,
                                            Helper.ToInterleaved(Angle),
                                            Helper.ToInterleaved(Position),
                                            (uint)BatchSize);
                }

                double CurrentLearningRate = MathHelper.Lerp(LearningRateStart,
                                                                LearningRateFinish,
                                                                iter / (float)NIterations);

                if (iter < 100)
                    CurrentLearningRate = MathHelper.Lerp(0, (float)CurrentLearningRate, iter / 99f);

                bool Twist = Rand.Next(2) == 0;

                model.Train(Twist ? ExtractedSource : ExtractedTarget,
                                    Twist ? ExtractedTarget : ExtractedSource,
                                    (float)CurrentLearningRate,
                                    out PredictedData,
                                    out Loss);

                Losses.Enqueue(Loss[0]);
                if (Losses.Count > 10)
                    Losses.Dequeue();

                TimeSpan TimeRemaining = Watch.Elapsed * (NIterations - 1 - iter);

                if (iter % 10 == 0 || iter == NIterations - 1)
                {
                    string ToWrite = $"{iter + 1}/{NIterations}, " +
                                        (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
                                        (iter > 10 ? $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " : "") +
                                        $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
                                        $"lr = {CurrentLearningRate:F6}" + (iter < 100 ? " (warm-up), " : ", ") +
                                        $"{GPU.GetFreeMemory(GPUNetwork)} MB free";

                    try
                    {
                        ClearCurrentConsoleLine();
                        Console.Write(ToWrite);
                    }
                    catch
                    {
                        // When we're outputting to a text file when launched on HPC cluster
                        Console.WriteLine(ToWrite);
                    }
                }

                if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                    throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                GPU.CheckGPUExceptions();
                Watch.Restart();
            }

            Watch.Stop();

            #region Cleanup

            ExtractedSource.Dispose();
            ExtractedTarget.Dispose();

            GPU.DestroyTexture(Texture1[0], TextureArray1[0]);
            GPU.DestroyTexture(Texture2[0], TextureArray2[0]);

            #endregion

            if (niterations > 0)
                Console.Write("\n");

            #endregion

            #region Inference

            //Console.WriteLine("Denoising combined map");

            Image MapCombined = MapHalf1Ori.GetCopy();
            MapCombined.Add(MapHalf2Ori);
            MapCombined.Multiply(0.5f);
            NoiseNet3DTorch.Denoise(MapCombined, new NoiseNet3DTorch[] { model });
            MapCombined.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
            denoisedCombined = MapCombined;

            //Console.WriteLine("Denoising half-map 1");
            NoiseNet3DTorch.Denoise(MapHalf1Ori, new NoiseNet3DTorch[] { model });
            MapHalf1Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
            denoised1 = MapHalf1Ori;

            //Console.WriteLine("Denoising half-map 2");
            NoiseNet3DTorch.Denoise(MapHalf2Ori, new NoiseNet3DTorch[] { model });
            MapHalf2Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
            denoised2 = MapHalf2Ori;

            #endregion
        }

        public static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
