using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public static class Logger
    {
        static readonly List<List<LogMessage>> Buffers = new List<List<LogMessage>>();
        static readonly Dictionary<Thread, int> ThreadMap = new Dictionary<Thread, int>();

        public static event MessageLoggedEvent MessageLogged;

        public static void SetBufferCount(int n)
        {
            while (Buffers.Count < n)
                Buffers.Add(new List<LogMessage>());
        }

        public static List<LogMessage> GetBuffer(int id)
        {
            if (id < 0 || id >= Buffers.Count)
                throw new Exception("No such buffer.");

            return Buffers[id];
        }

        public static void Write(LogMessage message, bool isGeneral = false)
        {
            int BufferID = 0;
            lock (ThreadMap)
                if (!isGeneral && ThreadMap.ContainsKey(Thread.CurrentThread))
                    BufferID = ThreadMap[Thread.CurrentThread];

            lock (Buffers[BufferID])
            {
                Buffers[BufferID].Add(message);
                MessageLogged?.Invoke(message, BufferID, Buffers[BufferID]);
            }
        }

        public static void MapThreadToBuffer(Thread thread, int bufferID)
        {
            lock (ThreadMap)
            {
                if (!ThreadMap.ContainsKey(thread))
                    ThreadMap.Add(thread, bufferID);
                else
                    ThreadMap[thread] = bufferID;
            }
        }

        public static void UnmapThread(Thread thread)
        {
            lock (ThreadMap)
            {
                if (ThreadMap.ContainsKey(thread))
                    ThreadMap.Remove(thread);
            }
        }
    }

    public delegate void MessageLoggedEvent(LogMessage message, int bufferID, List<LogMessage> buffer);

    public class LogMessage : WarpBase
    {
        private DateTime _Timestamp = DateTime.Now;
        public DateTime Timestamp
        {
            get { return _Timestamp; }
            set { if (value != _Timestamp) { _Timestamp = value; OnPropertyChanged(); } }
        }

        private object _Content = null;
        public object Content
        {
            get { return _Content; }
            set { if (value != _Content) { _Content = value; OnPropertyChanged(); } }
        }

        private string _GroupTitle = "";
        public string GroupTitle
        {
            get { return _GroupTitle; }
            set { if (value != _GroupTitle) { _GroupTitle = value; OnPropertyChanged(); } }
        }

        public LogMessage(object content, string groupTitle)
        {
            Content = content;
            GroupTitle = groupTitle;
            Timestamp = DateTime.Now;
        }

        public void Update(object content)
        {
            Timestamp = DateTime.Now;
            Content = content;
        }
    }
}
