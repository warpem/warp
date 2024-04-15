using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace EstimateWeights
{
    class EstimateWeights
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            VirtualConsole.AttachToConsole();

            Options Options = new Options();
            string WorkingDirectory;

            Dictionary<int, string> MaskOverrideNames = new Dictionary<int, string>();

            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";
            }

            Options.PopulationPath = Path.Combine(WorkingDirectory, Options.PopulationPath);

            if (!Options.ResolveFrames && !Options.ResolveItems && !Options.ResolveLocation && !Options.ResolveSources)
                throw new Exception("No dimension chosen to resolve over.");

            if (Options.ResolveLocation && (Options.ResolveFrames || Options.ResolveItems))
                throw new Exception("Spatial resolution cannot be combined with other dimensions in the same run. Please execute separately with different dimensions.");

            Population P = new Population(Options.PopulationPath);
            DataSource[] Sources = P.Sources.ToArray();

            float PixelSize = (float)P.Species[0].PixelSize;

            if (Options.DoReset)
            {
                Console.WriteLine("Resetting all weights... ");

                foreach (var source in Sources)
                {
                    source.RelativeBFactor = new float3(0, 0, 0);
                    source.RelativeWeight = 1;
                    source.Save();

                    Parallel.ForEach(source.Files.Values, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, item =>
                    {
                        Movie Item;

                        string ItemPath = Path.Combine(source.FolderPath, item);

                        if (item.Contains(".tomostar"))
                            Item = new TiltSeries(ItemPath);
                        else
                            Item = new Movie(ItemPath);

                        Item.GlobalBfactor = 0;
                        Item.GlobalWeight = 1;

                        Item.GridDoseBfacs = new CubicGrid(new int3(1));
                        Item.GridDoseBfacsAngle = new CubicGrid(new int3(1));
                        Item.GridDoseBfacsDelta = new CubicGrid(new int3(1));
                        Item.GridDoseWeights = new CubicGrid(new int3(1), new[] { 1f });

                        Item.GridLocationBfacs = new CubicGrid(new int3(1));
                        Item.GridLocationWeights = new CubicGrid(new int3(1), new[] { 1f });

                        Item.SaveMeta();
                    });
                }

                Console.WriteLine("Done.");

                return;
            }

            if (!Options.ResolveSources)
            {
                foreach (var source in Sources)
                {
                    string RefinementDirectory = Path.Combine(Helper.PathToFolder(Options.PopulationPath), "refinement_temp", source.Name);
                    //string RefinementDirectory = Path.Combine(Helper.PathToFolder(Options.PopulationPath), "refinement_temp", "funwith36", "ctf_distortion");

                    List<Movie> ItemsWithData = new List<Movie>();

                    Console.Write("Discovering items with data...");

                    List<string> FilesInFolder = Directory.EnumerateFiles(RefinementDirectory, "*.*").Select(f => Helper.PathToNameWithExtension(f)).ToList();

                    Movie[] AllItems = new Movie[source.Files.Count];
                    string[] AllNames = source.Files.Values.ToArray();
                    int NItemsDiscovered = 0;
                    Parallel.For(0, AllNames.Length, new ParallelOptions() { MaxDegreeOfParallelism = 4 }, i =>
                    {
                        string item = AllNames[i];

                        if (Options.DoTiltFrames)
                        {
                            bool SomethingMissing = false;
                            string ItemPath = Path.Combine(source.FolderPath, item);
                            TiltSeries Series = new TiltSeries(ItemPath);

                            for (int t = 0; t < Series.NTilts; t++)
                            {
                                if (!File.Exists(Path.Combine(RefinementDirectory, $"{Helper.PathToName(item)}_tilt{t:D3}_fsc.mrc")))
                                {
                                    SomethingMissing = true;
                                    break;
                                }
                            }

                            if (SomethingMissing)
                                return;

                            AllItems[i] = Series;
                        }
                        else
                        {
                            if (Options.ResolveFrames || Options.ResolveItems)
                            {
                                if (!FilesInFolder.Contains(Helper.PathToName(item) + "_fsc.mrc"))
                                    return;
                            }
                            else if (Options.ResolveLocation)
                            {
                                if (!FilesInFolder.Contains(Helper.PathToName(item) + "_fscparticles.mrc"))
                                    return;
                                if (!FilesInFolder.Contains(Helper.PathToName(item) + "_fscparticles.star"))
                                    return;
                            }

                            string ItemPath = Path.Combine(source.FolderPath, item);

                            if (item.Contains(".tomostar"))
                                AllItems[i] = new TiltSeries(ItemPath);
                            else
                                AllItems[i] = new Movie(ItemPath);
                        }

                        lock (AllItems)
                        {
                            NItemsDiscovered++;

                            VirtualConsole.ClearLastLine();
                            Console.Write($"Discovering items with data... {NItemsDiscovered}/{source.Files.Count}");
                        }
                    });
                    VirtualConsole.ClearLastLine();
                    Console.Write($"Discovering items with data... {NItemsDiscovered}/{source.Files.Count}");

                    Console.WriteLine();

                    ItemsWithData = AllItems.Where(v => v != null).ToList();

                    if (ItemsWithData.Count == 0)
                        Console.WriteLine($"No items with FSC data found for source {source.Name}!");

                    if (Options.DoTiltFrames)
                    {
                        for (int t = 0; t < ((TiltSeries)ItemsWithData[0]).NTilts; t++)
                        {
                            int2 Dims;
                            int MaxFrames = 1;
                            {
                                string FSCPath = Path.Combine(RefinementDirectory, $"{ItemsWithData[0].RootName}_tilt{t:D3}_fsc.mrc");
                                Image FSC = Image.FromFile(FSCPath);
                                Dims = new int2(FSC.Dims.Y);
                                MaxFrames = FSC.Dims.Z / 3;
                                FSC.Dispose();
                            }

                            List<float[]> AllAB = new List<float[]>();
                            List<float[]> AllA2 = new List<float[]>();
                            List<float[]> AllB2 = new List<float[]>();

                            foreach (var parentItem in ItemsWithData.Select(v => (TiltSeries)v))
                            {
                                string MoviePath = Path.Combine(parentItem.DataDirectoryName, parentItem.TiltMoviePaths[t]);
                                Movie item = new Movie(MoviePath);

                                {
                                    string FSCPath = Path.Combine(RefinementDirectory, $"{parentItem.RootName}_tilt{t:D3}_fsc.mrc");

                                    Image FSC = Image.FromFile(FSCPath);
                                    float[][] FSCData = FSC.GetHost(Intent.Read);

                                    for (int z = 0; z < FSCData.Length / 3; z++)
                                    {
                                        if (AllAB.Count <= z)
                                        {
                                            AllAB.Add(FSCData[z * 3 + 0]);
                                            AllA2.Add(FSCData[z * 3 + 1]);
                                            AllB2.Add(FSCData[z * 3 + 2]);
                                        }
                                        else
                                        {
                                            for (int i = 0; i < FSCData[0].Length; i++)
                                            {
                                                AllAB[z][i] += FSCData[z * 3 + 0][i];
                                                AllA2[z][i] += FSCData[z * 3 + 1][i];
                                                AllB2[z][i] += FSCData[z * 3 + 2][i];
                                            }
                                        }
                                    }

                                    FSC.Dispose();
                                }
                            }

                            int NItems = AllAB.Count;

                            Image CorrAB = new Image(AllAB.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                            Image CorrA2 = new Image(AllA2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                            Image CorrB2 = new Image(AllB2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);

                            Image CTFWeights = CorrAB.GetCopyGPU();
                            CTFWeights.Fill(1f);

                            (float[] ResultScales, float3[] ResultBfactors, _) = FSC.FitBFactors2D(CorrAB, CorrA2, CorrB2, CTFWeights, PixelSize, Options.MinResolution, Options.FitAnisotropy, 512, null);

                            CorrAB.Dispose();
                            CorrA2.Dispose();
                            CorrB2.Dispose();
                            CTFWeights.Dispose();

                            float MaxScale = MathHelper.Max(ResultScales);
                            ResultScales = ResultScales.Select(v => v / MaxScale).ToArray();

                            List<float> BfacsSorted = new List<float>(ResultBfactors.Select(v => v.X));
                            float MaxBfac = MathHelper.Max(BfacsSorted);
                            //float MaxBfac = MathHelper.Max(ResultBfactors.Select(v => v.X));
                            ResultBfactors = ResultBfactors.Select(v => new float3(Math.Min(0, v.X - MaxBfac), v.Y, v.Z)).ToArray();

                            List<float3> TableRows = new List<float3>();

                            foreach (var parentItem in ItemsWithData.Select(v => (TiltSeries)v))
                            {
                                string MoviePath = Path.Combine(parentItem.DataDirectoryName, parentItem.TiltMoviePaths[t]);
                                Movie item = new Movie(MoviePath);

                                int InThisItem = MaxFrames;

                                float[] Weights = ResultScales;
                                float[] Bfacs = ResultBfactors.Select(v => v.X).ToArray();
                                float[] BfacsDelta = ResultBfactors.Select(v => v.Y).ToArray();
                                float[] BfacsAngle = ResultBfactors.Select(v => v.Z).ToArray();

                                (int MaxIndex, float MaxB) = MathHelper.MaxElement(Bfacs);
                                TableRows.Add(new float3((float)item.CTF.Defocus,
                                                         Weights[MaxIndex],
                                                         MaxB));

                                item.GridDoseWeights = new CubicGrid(new int3(1, 1, InThisItem), Weights);

                                item.GridDoseBfacs = new CubicGrid(new int3(1, 1, InThisItem), Bfacs);
                                item.GridDoseBfacsDelta = new CubicGrid(new int3(1, 1, InThisItem), BfacsDelta);
                                item.GridDoseBfacsAngle = new CubicGrid(new int3(1, 1, InThisItem), BfacsAngle);

                                item.SaveMeta();
                            }
                        }
                    }
                    else if (Options.ResolveFrames || Options.ResolveItems)
                    {
                        int2 Dims;
                        int MaxFrames = 1;
                        {
                            string FSCPath = Path.Combine(RefinementDirectory, ItemsWithData[0].RootName + "_fsc.mrc");
                            Image FSC = Image.FromFile(FSCPath);
                            Dims = new int2(FSC.Dims.Y);
                            MaxFrames = FSC.Dims.Z / 3;
                            FSC.Dispose();
                        }

                        List<float[]> AllAB = new List<float[]>();
                        List<float[]> AllA2 = new List<float[]>();
                        List<float[]> AllB2 = new List<float[]>();
                        if (!Options.ResolveItems)
                        {
                            AllAB = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], MaxFrames));
                            AllA2 = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], MaxFrames));
                            AllB2 = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], MaxFrames));
                        }
                        else if (!Options.ResolveFrames)
                        {
                            AllAB = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], ItemsWithData.Count));
                            AllA2 = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], ItemsWithData.Count));
                            AllB2 = new List<float[]>(Helper.ArrayOfFunction(i => new float[Dims.ElementsFFT()], ItemsWithData.Count));
                        }

                        Console.Write("Loading correlation data...");

                        int NDone = 0;
                        foreach (var item in ItemsWithData)
                        {
                            string FSCPath = Path.Combine(RefinementDirectory, item.RootName + "_fsc.mrc");

                            Image FSC = Image.FromFile(FSCPath);
                            float[][] FSCData = FSC.GetHost(Intent.Read);

                            for (int z = 0; z < FSCData.Length / 3; z++)
                            {
                                if (Options.ResolveItems && Options.ResolveFrames)
                                {
                                    AllAB.Add(FSCData[z * 3 + 0]);
                                    AllA2.Add(FSCData[z * 3 + 1]);
                                    AllB2.Add(FSCData[z * 3 + 2]);
                                }
                                else
                                {
                                    int s = 0;
                                    if (!Options.ResolveItems)
                                        s = z;
                                    else if (!Options.ResolveFrames)
                                        s = NDone;
                                    else
                                        throw new Exception("Shouldn't be here");

                                    for (int i = 0; i < FSCData[0].Length; i++)
                                    {
                                        AllAB[s][i] += FSCData[z * 3 + 0][i];
                                        AllA2[s][i] += FSCData[z * 3 + 1][i];
                                        AllB2[s][i] += FSCData[z * 3 + 2][i];
                                    }
                                }
                            }

                            FSC.Dispose();
                            NDone++;
                            VirtualConsole.ClearLastLine();
                            Console.Write($"Loading correlation data... {NDone}/{ItemsWithData.Count}");
                        }

                        Console.WriteLine();

                        int NItems = AllAB.Count;

                        Image CorrAB = new Image(AllAB.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                        Image CorrA2 = new Image(AllA2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                        Image CorrB2 = new Image(AllB2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);

                        //Image CTFWeights = CorrAB.GetCopyGPU();
                        //CTFWeights.Fill(1f);

                        (float[] ResultScales, float3[] ResultBfactors, Image SumCorr) = FSC.FitBFactors2D(CorrAB, CorrA2, CorrB2, null, PixelSize, Options.MinResolution, Options.FitAnisotropy, 512, (progress) =>
                        {
                            VirtualConsole.ClearLastLine();
                            Console.Write($"Fitting... {(progress * 100).ToString("F4")} %");
                        });

                        Console.WriteLine();

                        List<float> ScalesSorted = ResultScales.Where((v, i) => ResultBfactors[i].X > -30).ToList();
                        ScalesSorted.Sort();
                        float MaxScale = Options.ResolveItems ? ScalesSorted[(int)(ScalesSorted.Count * 0.997f)] : ScalesSorted.Last();
                        ResultScales = ResultScales.Select(v => Math.Min(1, v / MaxScale)).ToArray();

                        List<float> BfacsSorted = Helper.ArrayOfSequence(0, ResultBfactors.Length, 1).Where(i => ResultScales[i] > 0.7).Select(i => ResultBfactors[i].X + Math.Abs(ResultBfactors[i].Y)).ToList();
                        BfacsSorted.Sort();
                        float MaxBfac = Options.ResolveItems ? BfacsSorted[(int)(BfacsSorted.Count * 0.997f)] : BfacsSorted.Last();
                        //float MaxBfac = MathHelper.Max(ResultBfactors.Select(v => v.X));
                        ResultBfactors = ResultBfactors.Select(v => new float3(Math.Min(0, v.X - MaxBfac), v.Y, v.Z)).ToArray();

                        SumCorr.WriteMRC(Path.Combine(RefinementDirectory, $"{source.Name}_sumcorr.mrc"), true);


                        NDone = 0;
                        int NFramesDone = 0;

                        foreach (var item in ItemsWithData)
                        {
                            int InThisItem = MaxFrames;
                            if (item.GetType() == typeof(TiltSeries))
                                InThisItem = ((TiltSeries)item).NTilts;

                            if (Options.ResolveFrames)
                            {
                                float[] Weights = ResultScales.Skip(NFramesDone).Take(InThisItem).ToArray();
                                float[] Bfacs = ResultBfactors.Skip(NFramesDone).Take(InThisItem).Select(v => v.X).ToArray();
                                float[] BfacsDelta = ResultBfactors.Skip(NFramesDone).Take(InThisItem).Select(v => v.Y).ToArray();
                                float[] BfacsAngle = ResultBfactors.Skip(NFramesDone).Take(InThisItem).Select(v => v.Z).ToArray();

                                item.GridDoseWeights = new CubicGrid(new int3(1, 1, InThisItem), Weights);

                                item.GridDoseBfacs = new CubicGrid(new int3(1, 1, InThisItem), Bfacs);
                                item.GridDoseBfacsDelta = new CubicGrid(new int3(1, 1, InThisItem), BfacsDelta);
                                item.GridDoseBfacsAngle = new CubicGrid(new int3(1, 1, InThisItem), BfacsAngle);

                                if (Options.ResolveItems)
                                    NFramesDone += InThisItem;
                            }
                            else if (Options.ResolveItems)
                            {
                                item.GlobalBfactor = ResultBfactors[NFramesDone].X;
                                item.GlobalWeight = ResultScales[NFramesDone];

                                NFramesDone++;
                            }

                            item.SaveMeta();

                            NDone++;
                            VirtualConsole.ClearLastLine();
                            Console.Write($"Saving metadata... {NDone}/{ItemsWithData.Count}");
                        }
                        Console.WriteLine();
                    }
                    else if (Options.ResolveLocation)
                    {
                        List<List<Movie>> ItemGroups = new List<List<Movie>>();
                        foreach (var item in ItemsWithData)
                            ItemGroups.Add(new List<Movie>() { item });

                        int FSCLength = 0;
                        {
                            string FSCPath = Path.Combine(RefinementDirectory, ItemsWithData[0].RootName + "_fscparticles.mrc");
                            Image FSC = Image.FromFile(FSCPath);
                            FSCLength = FSC.Dims.X;
                            FSC.Dispose();
                        }

                        int NGroupsDone = 0;
                        foreach (var itemGroup in ItemGroups)
                        {
                            int2 DimsGrid = new int2(Options.GridWidth, Options.GridHeight);
                            float2 GridMultiplier = new float2(DimsGrid - 1);
                            float3[][] AllFSCs = Helper.ArrayOfFunction(i => new float3[FSCLength], (int)DimsGrid.Elements());
                            float[][] AllCTFs = Helper.ArrayOfFunction(i => Helper.ArrayOfConstant(1f, FSCLength), (int)DimsGrid.Elements());

                            foreach (var item in itemGroup)
                            {
                                string FSCPath = Path.Combine(RefinementDirectory, item.RootName + "_fscparticles.mrc");
                                string StarPath = Path.Combine(RefinementDirectory, item.RootName + "_fscparticles.star");

                                Image FSC = Image.FromFile(FSCPath);
                                float[] FSCData = FSC.GetHost(Intent.Read)[0];

                                Star TableIn = new Star(StarPath);
                                float2[] Coords = TableIn.GetFloat2("wrpNormCoordinateX", "wrpNormCoordinateY");

                                if (Coords.Length * 3 != FSC.Dims.Y)
                                    throw new Exception($"Number of particles does not match number of FSC lines");

                                for (int p = 0; p < Coords.Length; p++)
                                {
                                    int2 CoordsRounded = new int2((int)Math.Round(Coords[p].X * GridMultiplier.X),
                                                                  (int)Math.Round(Coords[p].Y * GridMultiplier.Y));
                                    int FSCID = CoordsRounded.Y * DimsGrid.X + CoordsRounded.X;
                                    float3[] LocalFSC = AllFSCs[FSCID];

                                    for (int i = 0; i < FSCLength; i++)
                                        LocalFSC[i] += new float3(FSCData[(p * 3 + 0) * FSCLength + i],
                                                                  FSCData[(p * 3 + 1) * FSCLength + i],
                                                                  FSCData[(p * 3 + 2) * FSCLength + i]);
                                }

                                for (int i = 0; i < AllFSCs.Length; i++)
                                {
                                    if (AllFSCs[i][10].LengthSq() > 0)
                                        continue;

                                    int X = i % DimsGrid.X;
                                    int Y = i / DimsGrid.Y;

                                    int Closest = i;
                                    int ClosestDist = int.MaxValue;
                                    for (int j = 0; j < AllFSCs.Length; j++)
                                    {
                                        if (AllFSCs[j][10].LengthSq() > 0)
                                        {
                                            int XX = j % DimsGrid.X;
                                            int YY = j / DimsGrid.X;

                                            XX -= X;
                                            YY -= Y;
                                            int Dist = XX * XX + YY * YY;
                                            if (Dist < ClosestDist)
                                            {
                                                ClosestDist = Dist;
                                                Closest = j;
                                            }
                                        }
                                    }

                                    AllFSCs[i] = AllFSCs[Closest].ToArray();
                                }

                                FSC.Dispose();
                            }

                            try
                            {
                                float2[] PositionFits = FSC.FitBFactors(AllFSCs, AllCTFs, PixelSize, Options.MinResolution);

                                float MaxWeight = MathHelper.Max(PositionFits.Select(v => v.X));
                                float MaxBfactor = MathHelper.Max(PositionFits.Select(v => v.Y));

                                PositionFits = PositionFits.Select(v => new float2(v.X / MaxWeight, v.Y - MaxBfactor)).ToArray();

                                //Image Fitted = new Image(new[] { PositionFits.Select(v => v.X).ToArray(), PositionFits.Select(v => v.Y).ToArray() }, new int3(DimsGrid.X, DimsGrid.Y, 2));
                                //Fitted.WriteMRC($"d_fitted_{NGroupsDone}.mrc", true);

                                foreach (var item in itemGroup)
                                {
                                    item.GridLocationWeights = new CubicGrid(new int3(DimsGrid.X, DimsGrid.Y, 1), PositionFits.Select(v => v.X).ToArray());
                                    item.GridLocationBfacs = new CubicGrid(new int3(DimsGrid.X, DimsGrid.Y, 1), PositionFits.Select(v => v.Y).ToArray());

                                    item.SaveMeta();
                                }
                            }
                            catch
                            {
                                Console.WriteLine("Grid too fine for " + itemGroup.First().Name);
                            }

                            NGroupsDone++;
                        }
                    }

                    source.Save();
                }
            }
            else
            {
                string[] SourcePaths = Sources.Select(s => Path.Combine(Helper.PathToFolder(Options.PopulationPath), "refinement_temp", s.Name, $"{s.Name}_sumcorr.mrc")).Where(p => File.Exists(p)).ToArray();

                int2 Dims;
                int MaxFrames = 1;
                {
                    Image FSC = Image.FromFile(SourcePaths.First());
                    Dims = new int2(FSC.Dims.Y);
                    MaxFrames = FSC.Dims.Z / 3;
                    FSC.Dispose();
                }

                List<float[]> AllAB = new List<float[]>();
                List<float[]> AllA2 = new List<float[]>();
                List<float[]> AllB2 = new List<float[]>();

                Console.Write("Loading correlation data...");

                int NDone = 0;
                foreach (var path in SourcePaths)
                {
                    Image FSC = Image.FromFile(path);
                    float[][] FSCData = FSC.GetHost(Intent.Read);

                    AllAB.Add(FSCData[0]);
                    AllA2.Add(FSCData[1]);
                    AllB2.Add(FSCData[2]);

                    FSC.Dispose();
                    NDone++;
                    VirtualConsole.ClearLastLine();
                    Console.Write($"Loading correlation data... {NDone}/{SourcePaths.Length}");
                }

                Console.WriteLine();

                int NItems = AllAB.Count;

                Image CorrAB = new Image(AllAB.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                Image CorrA2 = new Image(AllA2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);
                Image CorrB2 = new Image(AllB2.ToArray(), new int3(Dims.X, Dims.Y, NItems), true);

                //Image CTFWeights = CorrAB.GetCopyGPU();
                //CTFWeights.Fill(1f);

                (float[] ResultScales, float3[] ResultBfactors, _) = FSC.FitBFactors2D(CorrAB, CorrA2, CorrB2, null, PixelSize, Options.MinResolution, Options.FitAnisotropy, 512, (progress) =>
                {
                    VirtualConsole.ClearLastLine();
                    Console.Write($"Fitting... {(progress * 100).ToString("F4")} %");
                });

                Console.WriteLine();

                List<float> ScalesSorted = ResultScales.ToList();
                ScalesSorted.Sort();
                float MaxScale = ScalesSorted.Last();
                ResultScales = ResultScales.Select(v => Math.Min(1, v / MaxScale)).ToArray();

                List<float> BfacsSorted = ResultBfactors.Select(v => v.X).ToList();
                BfacsSorted.Sort();
                float MaxBfac = BfacsSorted.Last();
                //float MaxBfac = MathHelper.Max(ResultBfactors.Select(v => v.X));
                ResultBfactors = ResultBfactors.Select(v => new float3(Math.Min(0, v.X - MaxBfac), v.Y, v.Z)).ToArray();

            }
        }
    }
}
