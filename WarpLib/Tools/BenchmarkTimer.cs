using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class BenchmarkTimer
    {
        private List<long> TimeMilliseconds = new List<long>();
        private List<long> TimeTicks = new List<long>();
        private List<DateTime> TimeStamps = new List<DateTime>();

        private string _Name = "";
        public string Name => _Name;

        public int NItems => TimeMilliseconds.Count;

        public BenchmarkTimer(string name)
        {
            _Name = name;
        }

        public (Stopwatch Watch, DateTime Stamp) Start()
        {
            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            return (Watch, DateTime.Now);
        }

        public void Finish((Stopwatch Watch, DateTime Stamp) timer)
        {
            timer.Watch.Stop();

            lock (TimeMilliseconds)
                TimeMilliseconds.Add(timer.Watch.ElapsedMilliseconds);
            lock (TimeTicks)
                TimeTicks.Add(timer.Watch.ElapsedTicks);
            lock (TimeStamps)
                TimeStamps.Add(timer.Stamp);
        }

        public void Clear()
        {
            lock (TimeMilliseconds)
                TimeMilliseconds.Clear();
            lock (TimeTicks)
                TimeTicks.Clear();
            lock (TimeStamps)
                TimeStamps.Clear();
        }

        public float GetAverageMilliseconds(int n)
        {
            if (TimeMilliseconds.Count == 0)
                return 0;

            lock (TimeMilliseconds)
            {
                if (n <= 0)
                    n = TimeMilliseconds.Count;
                n = Math.Min(n, TimeMilliseconds.Count);

                return (float)(TimeMilliseconds.Skip(TimeMilliseconds.Count - n).Sum() / (double)n);
            }
        }

        public float GetAverageTicks(int n)
        {
            if (TimeTicks.Count == 0)
                return 0;

            lock (TimeTicks)
            {
                if (n <= 0)
                    n = TimeTicks.Count;
                n = Math.Min(n, TimeTicks.Count);

                return (float)(TimeTicks.Skip(TimeTicks.Count - n).Sum() / (double)n);
            }
        }

        public float GetAverageMillisecondsConcurrent(int n)
        {
            if (TimeStamps.Count == 0)
                return 0;

            lock (TimeStamps)
            {
                if (n <= 0)
                    n = TimeStamps.Count;
                n = Math.Min(n, TimeStamps.Count);

                TimeSpan Diff = DateTime.Now - TimeStamps[TimeStamps.Count - n];
                return (float)Diff.TotalMilliseconds / n;
            }
        }

        public float GetPerSecondConcurrent(int n)
        {
            float AverageMinutes = GetAverageMillisecondsConcurrent(n) / 1000;

            return 1 / AverageMinutes;
        }
    }
}
