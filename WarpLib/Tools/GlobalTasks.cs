using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class GlobalTasks
    {
        static readonly List<Task> AllTasks = new List<Task>();

        public static void Add(Task task)
        {
            lock (AllTasks)
                AllTasks.Add(task);
        }

        public static void WaitAll()
        {
            Task[] Tasks;
            lock (AllTasks)
            {
                Tasks = AllTasks.ToArray();
                AllTasks.Clear();
            }

            Task.WaitAll(Tasks);
        }
    }
}
