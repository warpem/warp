using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools
{
    public static class VirtualConsole
    {
        static readonly List<LogEntry> Lines = new List<LogEntry>();
        public static readonly TextWriter Out = new VirtualTextWriter(Lines, LogEntryType.Regular);
        public static readonly TextWriter Error = new VirtualTextWriter(Lines, LogEntryType.Error);

        static readonly TextWriter SystemOut;
        static readonly TextWriter SystemError;

        public static string FileOutputPath = null;

        static bool IsAttached = false;
        static bool IsOutputPiped = false;
        public static bool IsSilent = false;

        static VirtualConsole()
        {
            ((VirtualTextWriter)Out).LineWritten += OutLineWritten;
            ((VirtualTextWriter)Out).Written += OutWritten;
            ((VirtualTextWriter)Error).LineWritten += ErrorLineWritten;
            ((VirtualTextWriter)Error).Written += ErrorWritten;
            
            IsOutputPiped = Console.IsOutputRedirected;

            SystemOut = Console.Out;
            SystemError = Console.Error;
        }

        public static int LineCount => Lines.Count;

        public static List<LogEntry> GetAllLines()
        {
            lock (Lines)
                return Lines.ToList();
        }

        public static List<LogEntry> GetLastNLines(int n)
        {
            lock (Lines)
                return Lines.Skip(Math.Max(0, Lines.Count - n)).ToList();
        }

        public static List<LogEntry> GetFirstNLines(int n)
        {
            lock (Lines)
                return Lines.Take(n).ToList();
        }

        public static List<LogEntry> GetLinesRange(int start, int end)
        {
            lock (Lines)
                return Lines.Skip(Math.Max(0, start)).Take(Math.Max(0, end - start)).ToList();
        }

        public static void ClearLastLine()
        {
            lock (Lines)
            {
                LogEntry Last = Lines[Lines.Count - 1];
                Lines[Lines.Count - 1] = Last with { Message = "" };

                UpdateFileOutput();
            }

            if (IsAttached && !IsSilent)
            {
                if (!IsOutputPiped && Console.WindowWidth > 1)
                {
                    int currentLineCursor = Console.CursorTop;
                    Console.SetCursorPosition(0, Console.CursorTop);
                    Console.Write(new string(' ', Console.WindowWidth - 2));
                    Console.SetCursorPosition(0, currentLineCursor);
                }
                else
                {
                    Console.Write("\r");
                }
            }
        }

        public static void ClearAll()
        {
            lock (Lines)
                Lines.Clear();

            if (IsAttached && !IsSilent)
                Console.Clear();
        }

        public static void AttachToConsole()
        {
            if (!IsAttached)
            {
                IsAttached = true;

                Console.SetOut(Out);
                Console.SetError(Error);
            }
        }

        public static void WriteToFile(string path)
        {
            using (TextWriter writer = File.CreateText(path))
            {
                lock (Lines)
                {
                    foreach (LogEntry line in Lines)
                        writer.WriteLine(line.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff") + " " + line.Message);
                }
            }
        }

        private static void UpdateFileOutput()
        {
            if (!string.IsNullOrEmpty(FileOutputPath))
                try
                {
                    using (TextWriter writer = File.CreateText(FileOutputPath))
                        foreach (LogEntry line in Lines)
                            writer.WriteLine(line.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff") + " " + line.Message);
                }
                catch { }
        }

        private static void OutLineWritten(string value)
        {
            if (IsAttached &&  !IsSilent)
                SystemOut.WriteLine(value);

            UpdateFileOutput();
        }

        private static void OutWritten(string value)
        {
            if (IsAttached && !IsSilent)
                SystemOut.Write(value);
        }

        private static void ErrorLineWritten(string value)
        {
            if (IsAttached && !IsSilent)
                SystemError.WriteLine(value);

            UpdateFileOutput();
        }

        private static void ErrorWritten(string value)
        {
            if (IsAttached && !IsSilent)
                SystemError.Write(value);
        }

        private static string EntryTypeToString(LogEntryType type)
        {
            return type switch 
            { 
                LogEntryType.Regular => "", 
                LogEntryType.Error => "Error", 
                _ => "Unknown" 
            };
        }
    }

    public record LogEntry
    {
        [JsonPropertyName("message")]
        public string Message { get; init; }

        [JsonPropertyName("timestamp")]
        public DateTime Timestamp { get; init; }

        [JsonPropertyName("type")]
        public LogEntryType Type { get; init; }

        public LogEntry(string message, DateTime timestamp, LogEntryType type)
        {
            Message = message;
            Timestamp = timestamp;
            Type = type;
        }
    };

    public enum LogEntryType
    {
        Regular = 0,
        Error = 1
    }

    public class VirtualTextWriter : TextWriter
    {
        readonly List<LogEntry> Lines;
        readonly LogEntryType Type;

        public event Action<string> Written;
        public event Action<string> LineWritten;

        public VirtualTextWriter(List<LogEntry> linesBuffer, LogEntryType type)
        {
            Lines = linesBuffer;
            Type = type;
        }

        public override Encoding Encoding => Encoding.UTF8;

        public override void Write(char value)
        {
            Write(value.ToString());
        }

        public override void Write(string value)
        {
            lock (Lines)
            {
                if (!Lines.Any())
                    Lines.Add(new LogEntry("", DateTime.Now, LogEntryType.Regular));

                LogEntry Last = Lines[Lines.Count - 1];
                Lines[Lines.Count - 1] = Last with { Message = Last.Message + value };

                Written?.Invoke(value);
            }
        }

        public override void WriteLine(string value)
        {
            string[] Parts = value.Split('\n');

            lock (Lines)
            {
                foreach (string part in Parts)
                    Lines.Add(new LogEntry(part, DateTime.UtcNow, Type));

                LineWritten?.Invoke(value);
            }
        }
    }
}
