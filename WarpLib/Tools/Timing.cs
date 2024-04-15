using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class Timing
    {
        static Dictionary<string, Stopwatch> Timers = new Dictionary<string, Stopwatch>();
        static Dictionary<string, List<double>> Measurements = new Dictionary<string, List<double>>();

        public static void Start(string name)
        {
            Stopwatch Timer = new Stopwatch();

            if (Timers.ContainsKey(name))
            {
                Timers[name] = Timer;
            }
            else
            {
                Timers.Add(name, Timer);
            }

            Timer.Start();
        }

        public static void Finish(string name)
        {
            if (Timers.ContainsKey(name))
            {
                Stopwatch Timer = Timers[name];
                Timer.Stop();

                if (!Measurements.ContainsKey(name))
                    Measurements.Add(name, new List<double>());

                Measurements[name].Add((double)Timer.ElapsedTicks / Stopwatch.Frequency * 1000);
            }
        }

        public static void PrintMeasurements()
        {
            Console.WriteLine("");

            foreach (var pair in Measurements)
            {
                double Mean = pair.Value.Sum() / pair.Value.Count;
                Console.WriteLine($"{pair.Key}: {Mean:F3} ms per block, {pair.Value.Sum():F3} ms overall");
            }

            Console.WriteLine("");
        }
    }
}
