using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class MallocCache
    {
        static LinkedList<float[]> CacheFloat = new LinkedList<float[]>();
        static LinkedList<int[]> CacheInt = new LinkedList<int[]>();

        public static float[] GetFloat(int elements, bool exactSize = false, bool initialize = false, float initValue = 0) 
        {
            float[] Result = null;
            bool InitZeros = false;

            lock (CacheFloat)
            {
                if (exactSize)
                {
                    foreach (var item in CacheFloat)
                        if (item.Length == elements)
                        {
                            Result = item;
                            break;
                        }
                }
                else
                {
                    foreach (var item in CacheFloat)
                        if (item.Length >= elements)
                        {
                            Result = item;
                            break;
                        }
                }

                if (Result != null)
                    CacheFloat.Remove(Result);
            }

            if (Result == null)
            {
                Result = new float[elements];
                InitZeros = true;
            }

            if (initialize && !(initValue == 0 && InitZeros))
                for (int i = 0; i < Result.Length; i++)
                    Result[i] = initValue;

            return Result;
        }

        public static void ReturnFloat(float[] array)
        {
            lock (CacheFloat)
            {
                float[] BestPosition = null;
                foreach (var item in CacheFloat)
                    if (item.Length < array.Length)
                        BestPosition = item;
                    else
                        break;

                if (BestPosition == null)
                    CacheFloat.AddFirst(array);
                else
                    CacheFloat.AddAfter(CacheFloat.Find(BestPosition), array);
            }
        }

        public static int[] GetInt(int elements, bool exactSize = false, bool initialize = false, int initValue = 0)
        {
            int[] Result = null;
            bool InitZeros = false;

            lock (CacheInt)
            {
                if (exactSize)
                {
                    foreach (var item in CacheInt)
                        if (item.Length == elements)
                        {
                            Result = item;
                            break;
                        }
                }
                else
                {
                    foreach (var item in CacheInt)
                        if (item.Length >= elements)
                        {
                            Result = item;
                            break;
                        }
                }

                if (Result != null)
                    CacheInt.Remove(Result);
            }

            if (Result == null)
            {
                Result = new int[elements];
                InitZeros = true;
            }

            if (initialize && !(initValue == 0 && InitZeros))
                for (int i = 0; i < Result.Length; i++)
                    Result[i] = initValue;

            return Result;
        }

        public static void ReturnInt(int[] array)
        {
            lock (CacheInt)
            {
                int[] BestPosition = null;
                foreach (var item in CacheInt)
                    if (item.Length < array.Length)
                        BestPosition = item;
                    else
                        break;

                if (BestPosition == null)
                    CacheInt.AddFirst(array);
                else
                    CacheInt.AddAfter(CacheInt.Find(BestPosition), array);
            }
        }

        public static void Flush()
        {
            lock (CacheFloat)
            {
                CacheFloat.Clear();
            }

            lock (CacheInt)
            {
                CacheInt.Clear();
            }
        }
    }
}
