using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class FFTPlanCache
    {
        private static Dictionary<string, int> Cache = new Dictionary<string, int>();
        private static int MaxSize = 4;

        // Lock object for thread-safety
        private static readonly object LockObject = new object();

        public static int GetFFTPlan(int3 dims, int batch) => GetOrCreatePlan(dims, batch, true);

        public static int GetIFFTPlan(int3 dims, int batch) => GetOrCreatePlan(dims, batch, false);

        private static int GetOrCreatePlan(int3 dims, int batch, bool isForward)
        {
            string Key = GenerateKey(dims, batch, isForward);

            lock (LockObject)
            {
                if (Cache.ContainsKey(Key))
                    return Cache[Key];

                int NewPlan = isForward
                    ? GPU.CreateFFTPlan(dims, (uint)batch)
                    : GPU.CreateIFFTPlan(dims, (uint)batch);

                while (Cache.Count > MaxSize)
                {
                    var OldKey = Cache.Keys.First();
                    var OldPlan = Cache[OldKey];
                    GPU.DestroyFFTPlan(OldPlan);
                    Cache.Remove(OldKey);
                }

                Cache.Add(Key, NewPlan);

                return NewPlan;
            }
        }

        public static void RemoveAndDestroyPlan(int plan)
        {
            lock (LockObject)
            {
                if (!Cache.ContainsValue(plan))
                    return;

                var Pair = Cache.FirstOrDefault(x => x.Value == plan);
                GPU.DestroyFFTPlan(plan);
                Cache.Remove(Pair.Key);
            }
        }

        public static void Clear()
        {
            lock (LockObject)
            {
                foreach (var plan in Cache.Values)
                    GPU.DestroyFFTPlan(plan);

                Cache.Clear();
            }
        }

        public static void SetSize(int size)
        {
            lock (LockObject)
            {
                MaxSize = size;

                while (Cache.Count > MaxSize)
                {
                    var OldKey = Cache.Keys.First();
                    var OldPlan = Cache[OldKey];
                    GPU.DestroyFFTPlan(OldPlan);
                    Cache.Remove(OldKey);
                }
            }
        }

        private static string GenerateKey(int3 dimensions, int batch, bool isForward)
        {
            return $"{dimensions.X}-{dimensions.Y}-{dimensions.Z}-{batch}-{isForward}";
        }
    }
}
