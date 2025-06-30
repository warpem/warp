using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;
using Accord;
using Warp.Tools;
using ZLinq;

namespace Warp
{
    public class Star
    {
        protected Dictionary<string, int> NameMapping = new Dictionary<string, int>();
        protected List<string[]> Rows = new List<string[]>();

        public int RowCount => Rows.Count;
        public int ColumnCount => NameMapping.Count;

        public Star(string path, string tableName = "", string[] onlyColumns = null, int nrows = -1, string rowFilterColumn = null, Func<string, bool> rowFilter = null)
        {
            using (SpanLineReader Reader = new SpanLineReader(path))
            {
                string Line;

                if (!string.IsNullOrEmpty(tableName))
                {
                    tableName = "data_" + tableName;
                    while((Line = Reader.ReadLine()) != null && !Line.StartsWith(tableName)) ;

                    if (Line == null)
                        throw new Exception($"Table {tableName} not found in {path}");
                }

                while((Line = Reader.ReadLine()) != null && !Line.Contains("loop_")) ;

                while(true)
                {
                    Line = Reader.ReadLine();

                    if (Line == null)
                        break;
                    if (Line.Length == 0)
                        continue;
                    if (Line[0] != '_')
                        break;

                    string[] Parts = Line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    string ColumnName = Parts[0].Substring(1);
                    int ColumnIndex = Parts.Length > 1 ? int.Parse(Parts[1].Substring(1)) - 1 : NameMapping.Count;
                    NameMapping.Add(ColumnName, ColumnIndex);
                }
                
                int RowFilterColumnId = -1;
                if (rowFilter != null && !string.IsNullOrWhiteSpace(rowFilterColumn))
                {
                    if (!NameMapping.ContainsKey(rowFilterColumn))
                        throw new Exception($"Row filter column '{rowFilterColumn}' not found in {path}, table {tableName}");

                    RowFilterColumnId = NameMapping[rowFilterColumn];
                }

                if (onlyColumns == null)
                {
                    ReadOnlySpan<char> LineSpan = Line.AsSpan(); // First line has already been read

                    if (RowFilterColumnId < 0)
                        do
                        {
                            string[] Parts = Helper.StringSplitWhitespace(LineSpan, null, new string[NameMapping.Count]);
                            if (Parts.Length == NameMapping.Count)
                                Rows.Add(Parts);
                            else
                                break;

                            if (nrows > 0 && Rows.Count >= nrows)
                                break;
                        } while(Reader.TryReadLineSpan(out LineSpan));
                    else
                        do
                        {
                            string[] Parts = Helper.StringSplitWhitespace(LineSpan, null, new string[NameMapping.Count]);
                            if (Parts.Length == NameMapping.Count && rowFilter(Parts[RowFilterColumnId]))
                                Rows.Add(Parts);
                            else
                                break;

                            if (nrows > 0 && Rows.Count >= nrows)
                                break;
                        } while(Reader.TryReadLineSpan(out LineSpan));
                }
                else
                {
                    int OverallColumns = NameMapping.Count;

                    int[] OnlyPositions = new int[onlyColumns.Length];
                    for (int c = 0; c < onlyColumns.Length; c++)
                    {
                        if (!NameMapping.ContainsKey(onlyColumns[c]))
                            throw new Exception($"Column {onlyColumns[c]} requested, but not found in {path}, table {tableName}");
                        OnlyPositions[c] = NameMapping[onlyColumns[c]];
                    }

                    NameMapping.Clear();
                    foreach (var name in onlyColumns)
                        NameMapping.Add(name, NameMapping.Count);

                    bool[] ReadMask = new bool[OverallColumns];
                    foreach (var p in OnlyPositions)
                        ReadMask[p] = true;

                    ReadOnlySpan<char> LineSpan = Line.AsSpan(); // First line has already been read
                    string[] PartsBuffer = new string[OverallColumns];
                    if (RowFilterColumnId < 0)
                        do
                        {
                            string[] Parts = Helper.StringSplitWhitespace(LineSpan, ReadMask, PartsBuffer);
                            if (Parts.Length == OverallColumns)
                                Rows.Add(Helper.IndexedSubset(Parts, OnlyPositions));
                            else
                                break;

                            if (nrows > 0 && Rows.Count >= nrows)
                                break;
                        } while(Reader.TryReadLineSpan(out LineSpan));
                    else
                        do
                        {
                            string[] Parts = Helper.StringSplitWhitespace(LineSpan, ReadMask, PartsBuffer);
                            if (Parts.Length == OverallColumns && rowFilter(Parts[RowFilterColumnId]))
                                Rows.Add(Helper.IndexedSubset(Parts, OnlyPositions));
                            else
                                break;

                            if (nrows > 0 && Rows.Count >= nrows)
                                break;
                        } while(Reader.TryReadLineSpan(out LineSpan));
                }
            }
        }

        public Star(string[] columnNames)
        {
            foreach (string name in columnNames)
                NameMapping.Add(name, NameMapping.Count);
        }

        public Star(Star[] tables)
        {
            List<string> Common = new List<string>(tables[0].GetColumnNames());

            foreach (var table in tables)
                Common.RemoveAll(c => !table.HasColumn(c));

            foreach (string name in Common)
                NameMapping.Add(name, NameMapping.Count);

            foreach (var table in tables)
            {
                int[] ColumnIndices = Common.Select(c => table.GetColumnID(c)).ToArray();

                for (int r = 0; r < table.RowCount; r++)
                {
                    string[] Row = new string[Common.Count];
                    for (int c = 0; c < ColumnIndices.Length; c++)
                        Row[c] = table.GetRowValue(r, ColumnIndices[c]);

                    AddRow(Row);
                }
            }
        }

        public Star(float[] values, string nameColumn1)
        {
            AddColumn(nameColumn1, values.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float2[] values, string nameColumn1, string nameColumn2)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float3[] values, string nameColumn1, string nameColumn2, string nameColumn3)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn3, values.Select(v => v.Z.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float4[] values, string nameColumn1, string nameColumn2, string nameColumn3, string nameColumn4)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn3, values.Select(v => v.Z.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn4, values.Select(v => v.W.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(float5[] values, string nameColumn1, string nameColumn2, string nameColumn3, string nameColumn4, string nameColumn5)
        {
            AddColumn(nameColumn1, values.Select(v => v.X.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn2, values.Select(v => v.Y.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn3, values.Select(v => v.Z.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn4, values.Select(v => v.W.ToString(CultureInfo.InvariantCulture)).ToArray());
            AddColumn(nameColumn5, values.Select(v => v.V.ToString(CultureInfo.InvariantCulture)).ToArray());
        }

        public Star(string[][] columnValues, params string[] columnNames)
        {
            if (columnValues.Length != columnNames.Length)
                throw new DimensionMismatchException();

            for (int i = 0; i < columnNames.Length; i++)
                AddColumn(columnNames[i], columnValues[i]);
        }

        public Star()
        {
        }

        public static Dictionary<string, Star> FromMultitable(string path, IEnumerable<string> names)
        {
            return names.Where(name => ContainsTable(path, name)).ToDictionary(name => name, name => new Star(path, name));
        }

        public static Dictionary<string, Star> ReadAllTables(string path)
        {
            Dictionary<string, Star> result = new Dictionary<string, Star>();
            List<string> tableNames = new List<string>();

            // discover all table names in the file
            using TextReader reader = File.OpenText(path);
            string line;
            while((line = reader.ReadLine()) != null)
            {
                if (line.StartsWith("data_") && line.Length > "data_".Length)
                {
                    string tableName = line.Substring("data_".Length).Trim();
                    if (!string.IsNullOrEmpty(tableName))
                    {
                        tableNames.Add(tableName);
                    }
                }
            }

            // create Star instances for each table
            foreach (string tableName in tableNames)
            {
                try
                {
                    Star table = new Star(path, tableName);
                    result.Add(tableName, table);
                }
                catch(Exception ex)
                {
                    throw new Exception($"Error loading table '{tableName}' from {path}: {ex.Message}");
                }
            }

            return result;
        }

        public static bool IsMultiTable(string path)
        {
            bool IsMulti = false;

            using (TextReader Reader = File.OpenText(path))
            {
                string Line;
                while((Line = Reader.ReadLine()) != null)
                    if (Line.StartsWith("data_"))
                    {
                        if (Line.Replace("\n", "").Replace("\r", "").Length > "data_".Length)
                            IsMulti = true;
                        break;
                    }
            }

            return IsMulti;
        }

        public static bool ContainsTable(string path, string name)
        {
            bool Found = false;
            name = "data_" + name;

            using (TextReader Reader = File.OpenText(path))
            {
                string Line;
                while((Line = Reader.ReadLine()) != null)
                    if (Line.StartsWith(name))
                    {
                        Found = true;
                        break;
                    }
            }

            return Found;
        }


        public static (Star table, bool is3) LoadRelion3Particles(string path)
        {
            // Ceci n'est pas 3.0+?
            if (!IsMultiTable(path) || !ContainsTable(path, "optics"))
                return (new Star(path), false);

            Star TableOptics = new Star(path, "optics");

            int[] GroupIDs = TableOptics.GetColumn("rlnOpticsGroup").Select(s => int.Parse(s)).ToArray();

            if (TableOptics.HasColumn("rlnImagePixelSize"))
            {
                string[] PixelSizes = TableOptics.GetColumn("rlnImagePixelSize");

                TableOptics.RemoveColumn("rlnImagePixelSize");
                TableOptics.AddColumn("rlnDetectorPixelSize", PixelSizes);
                TableOptics.AddColumn("rlnMagnification", "10000");
            }

            string[][] OpticsGroups = new string[GroupIDs.Max() + 1][];
            string[] OpticsFieldNames = TableOptics.GetColumnNames().Where(n => n != "rlnOpticsGroup" && n != "rlnOpticsGroupName").ToArray();

            for (int r = 0; r < TableOptics.RowCount; r++)
            {
                List<string> OpticsFields = new List<string>();
                foreach (var columnName in OpticsFieldNames)
                    OpticsFields.Add(TableOptics.GetRowValue(r, columnName));

                OpticsGroups[GroupIDs[r]] = OpticsFields.ToArray();
            }

            Star TableParticles = new Star(path, "particles");

            foreach (var fieldName in OpticsFieldNames)
                if (TableParticles.HasColumn(fieldName))
                    TableParticles.RemoveColumn(fieldName);

            int[] ColumnOpticsGroupID = TableParticles.GetColumn("rlnOpticsGroup").Select(s => int.Parse(s)).ToArray();

            for (int iField = 0; iField < OpticsFieldNames.Length; iField++)
            {
                string[] NewColumn = new string[TableParticles.RowCount];

                for (int r = 0; r < ColumnOpticsGroupID.Length; r++)
                {
                    int GroupID = ColumnOpticsGroupID[r];
                    NewColumn[r] = OpticsGroups[GroupID][iField];
                }

                TableParticles.AddColumn(OpticsFieldNames[iField], NewColumn);
            }

            TableParticles.RemoveColumn("rlnOpticsGroup");

            return (TableParticles, true);
        }

        public static void SaveMultitable(string path, Dictionary<string, Star> tables)
        {
            bool WrittenFirst = false;
            foreach (var pair in tables)
            {
                pair.Value.Save(path, pair.Key, WrittenFirst);
                WrittenFirst = true;
            }
        }

        public virtual void Save(string path, string name = "", bool append = false)
        {
            using (TextWriter Writer = append ? File.AppendText(path) : File.CreateText(path))
            {
                if (append)
                    Writer.Write("\n\n\n");

                Writer.Write("\n");
                Writer.Write("data_" + name + "\n");
                Writer.Write("\n");
                Writer.Write("loop_\n");

                foreach (var pair in NameMapping)
                    Writer.Write($"_{pair.Key} #{pair.Value + 1}\n");

                int NThreads = 8;
                int[][] ColumnWidths = Helper.ArrayOfFunction(i => new int[ColumnCount], NThreads);
                Helper.ForCPU(0, Rows.Count, NThreads, null, (r, threadID) =>
                {
                    string[] row = Rows[r];
                    for (int i = 0; i < row.Length; i++)
                        ColumnWidths[threadID][i] = Math.Max(ColumnWidths[threadID][i], row[i].Length);
                }, null);
                for (int c = 0; c < ColumnCount; c++)
                for (int t = 1; t < NThreads; t++)
                    ColumnWidths[0][c] = Math.Max(ColumnWidths[0][c], ColumnWidths[t][c]);

                int RowLength = ColumnWidths[0].Select(v => v + 2).Sum();

                char[] RowBuilderBuffer = new char[RowLength + 1];
                Span<char> RowBuilder = new Span<char>(RowBuilderBuffer);
                foreach (var row in Rows)
                {
                    int Cursor = 0;
                    for (int i = 0; i < row.Length; i++)
                    {
                        int Whitespaces = 2 + ColumnWidths[0][i] - row[i].Length;
                        for (int w = 0; w < Whitespaces; w++)
                            RowBuilder[Cursor++] = ' ';

                        row[i].AsSpan().CopyTo(RowBuilder.Slice(Cursor));
                        Cursor += row[i].Length;
                    }

                    RowBuilder[Cursor] = '\n';
                    Writer.Write(RowBuilder);
                }
            }
        }

        public static int CountLines(string path)
        {
            int Result = 0;

            using (TextReader Reader = new StreamReader(File.OpenRead(path)))
            {
                string Line;

                while((Line = Reader.ReadLine()) != null && !Line.Contains("loop_")) ;

                while(true)
                {
                    Line = Reader.ReadLine();

                    if (Line == null)
                        break;
                    if (Line.Length == 0)
                        continue;
                    if (Line[0] != '_')
                        break;
                }

                do
                {
                    if (Line == null)
                        break;

                    if (Line.Length > 3)
                        Result++;
                } while((Line = Reader.ReadLine()) != null);
            }

            return Result;
        }

        public string[] GetColumn(string name)
        {
            if (!NameMapping.ContainsKey(name))
                return null;

            int Index = NameMapping[name];
            string[] Column = new string[Rows.Count];
            for (int i = 0; i < Rows.Count; i++)
                Column[i] = Rows[i][Index];

            return Column;
        }

        public string[] GetColumn(int id)
        {
            string[] Column = new string[Rows.Count];
            for (int i = 0; i < Rows.Count; i++)
                Column[i] = Rows[i][id];

            return Column;
        }

        public void SetColumn(string name, string[] values)
        {
            int Index = NameMapping[name];
            for (int i = 0; i < Rows.Count; i++)
                Rows[i][Index] = values[i];
        }

        public int GetColumnID(string name)
        {
            if (NameMapping.ContainsKey(name))
                return NameMapping[name];
            else
                return -1;
        }

        public List<string[]> GetAllRows()
        {
            return Rows;
        }

        public string GetRowValue(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValue(row, NameMapping[column]);
        }

        public string GetRowValue(int row, int column)
        {
            return Rows[row][column];
        }

        public float GetRowValueFloat(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValueFloat(row, NameMapping[column]);
        }

        public float GetRowValueFloat(int row, int column)
        {
            return float.Parse(Rows[row][column].Replace("inf", "Infinity")
                    .Replace("-nan", "NaN")
                    .Replace("nan", "NaN"),
                CultureInfo.InvariantCulture);
        }

        public int GetRowValueInt(int row, string column)
        {
            if (!NameMapping.ContainsKey(column))
                throw new Exception("Column does not exist.");
            if (row < 0 || row >= Rows.Count)
                throw new Exception("Row does not exist.");

            return GetRowValueInt(row, NameMapping[column]);
        }

        public int GetRowValueInt(int row, int column)
        {
            return int.Parse(Rows[row][column]);
        }

        public void SetRowValue(int row, string column, string value)
        {
            Rows[row][NameMapping[column]] = value;
        }

        public void SetRowValue(int row, int column, string value)
        {
            Rows[row][column] = value;
        }

        public void SetRowValue(int row, string column, float value)
        {
            Rows[row][NameMapping[column]] = value.ToString(CultureInfo.InvariantCulture);
        }

        public void SetRowValue(int row, string column, int value)
        {
            Rows[row][NameMapping[column]] = value.ToString();
        }

        public void ModifyAllValuesInColumn(string columnName, Func<string, string> f)
        {
            int ColumnID = GetColumnID(columnName);
            for (int r = 0; r < Rows.Count; r++)
                Rows[r][ColumnID] = f(Rows[r][ColumnID]);
        }

        public void ModifyAllValuesInColumn(string columnName, Func<string, int, string> f)
        {
            int ColumnID = GetColumnID(columnName);
            for (int r = 0; r < Rows.Count; r++)
                Rows[r][ColumnID] = f(Rows[r][ColumnID], r);
        }

        public bool HasColumn(string name)
        {
            return NameMapping.ContainsKey(name);
        }

        public void AddColumn(string name, string[] values)
        {
            int NewIndex = NameMapping.Count > 0 ? NameMapping.Select((v, k) => k).Max() + 1 : 0;
            NameMapping.Add(name, NewIndex);

            if (Rows.Count == 0)
                Rows = Helper.ArrayOfFunction(i => new string[NameMapping.Count], values.Length).ToList();

            if (Rows.Count != values.Length)
                throw new DimensionMismatchException();

            for (int i = 0; i < Rows.Count; i++)
            {
                string[] NewRow = new string[NameMapping.Count];

                Array.Copy(Rows[i], NewRow, NewRow.Length - 1);
                NewRow[NewRow.Length - 1] = values[i];

                Rows[i] = NewRow;
            }
        }

        public void AddColumn(string name, string defaultValue = "")
        {
            string[] EmptyValues = Helper.ArrayOfConstant(defaultValue, RowCount);

            AddColumn(name, EmptyValues);
        }

        public void RemoveColumn(string name)
        {
            int Index = NameMapping[name];

            for (int r = 0; r < Rows.Count; r++)
            {
                string[] OldRow = Rows[r];
                string[] NewRow = new string[OldRow.Length - 1];
                for (int i = 0, j = 0; i < OldRow.Length; i++)
                {
                    if (i == Index)
                        continue;
                    NewRow[j++] = OldRow[i];
                }

                Rows[r] = NewRow;
            }

            NameMapping.Remove(name);
            var BiggerNames = NameMapping.Where(vk => vk.Value > Index).Select(vk => vk.Key).ToArray();
            foreach (var biggerName in BiggerNames)
                NameMapping[biggerName] = NameMapping[biggerName] - 1;

            var KeyValuePairs = NameMapping.Select(vk => vk).ToList();
            KeyValuePairs.Sort((vk1, vk2) => vk1.Value.CompareTo(vk2.Value));
            NameMapping = new Dictionary<string, int>();
            foreach (var keyValuePair in KeyValuePairs)
                NameMapping.Add(keyValuePair.Key, keyValuePair.Value);
        }

        public string[] GetColumnNames()
        {
            return NameMapping.Select(pair => pair.Key).ToArray();
        }

        public string[] GetRow(int index)
        {
            return Rows[index];
        }

        public int CountRows(string columnName, Func<string, bool> match)
        {
            int ColumnID = GetColumnID(columnName);
            int Result = 0;

            for (int r = 0; r < Rows.Count; r++)
                if (match(Rows[r][ColumnID]))
                    Result++;

            return Result;
        }

        public int CountRows(string columnName1, string columnName2, Func<string, string, bool> match)
        {
            int ColumnID1 = GetColumnID(columnName1);
            int ColumnID2 = GetColumnID(columnName2);
            int Result = 0;

            for (int r = 0; r < Rows.Count; r++)
                if (match(Rows[r][ColumnID1], Rows[r][ColumnID2]))
                    Result++;

            return Result;
        }

        public int[] RowsWhere(string columnName, Func<string, bool> match)
        {
            int ColumnID = GetColumnID(columnName);
            List<int> Result = new List<int>();

            for (int r = 0; r < Rows.Count; r++)
                if (match(Rows[r][ColumnID]))
                    Result.Add(r);

            return Result.ToArray();
        }

        public int[] RowsWhere(string columnName1, string columnName2, Func<string, string, bool> match)
        {
            int ColumnID1 = GetColumnID(columnName1);
            int ColumnID2 = GetColumnID(columnName2);
            List<int> Result = new List<int>();

            for (int r = 0; r < Rows.Count; r++)
                if (match(Rows[r][ColumnID1], Rows[r][ColumnID2]))
                    Result.Add(r);

            return Result.ToArray();
        }

        public void AddRow(string[] row)
        {
            Rows.Add(row);
        }

        public void AddRow(IEnumerable<string[]> rows)
        {
            Rows.AddRange(rows);
        }

        public void RemoveAllRows()
        {
            Rows.Clear();
        }

        public void RemoveRows(int[] indices)
        {
            List<int> IndicesSorted = indices.ToList();
            IndicesSorted.Sort();

            for (int i = IndicesSorted.Count - 1; i >= 0; i--)
                Rows.RemoveAt(IndicesSorted[i]);
        }

        public void RemoveRowsWhere(string columnName, Func<string, bool> match)
        {
            RemoveRows(RowsWhere(columnName, match));
        }

        public void RemoveRowsWhere(string columnName1, string columnName2, Func<string, string, bool> match)
        {
            RemoveRows(RowsWhere(columnName1, columnName2, match));
        }

        public void SortByKey(string[] keys)
        {
            var SortedRows = Helper.ArrayOfFunction(i => (Rows[i], i), RowCount).ToList();
            SortedRows.Sort((a, b) => keys[a.i].CompareTo(keys[b.i]));

            Rows = SortedRows.Select(t => t.Item1).ToList();
        }

        public void SortByKey(string[] keys, Comparison<string> comparison)
        {
            var SortedRows = Helper.ArrayOfFunction(i => (Rows[i], i), RowCount).ToList();
            SortedRows.Sort((a, b) => comparison(keys[a.i], keys[b.i]));

            Rows = SortedRows.Select(t => t.Item1).ToList();
        }

        public Star CreateSubset(IEnumerable<int> rows)
        {
            Star Subset = new Star(GetColumnNames());
            foreach (var row in rows)
                Subset.AddRow(Rows[row].ToArray());

            return Subset;
        }

        public float3[] GetRelionCoordinates()
        {
            float[] X = HasColumn("rlnCoordinateX") ? GetColumn("rlnCoordinateX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Y = HasColumn("rlnCoordinateY") ? GetColumn("rlnCoordinateY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Z = HasColumn("rlnCoordinateZ") ? GetColumn("rlnCoordinateZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];

            return Helper.Zip(X, Y, Z);
        }

        public float3[] GetRelionOffsets()
        {
            float[] X = HasColumn("rlnOriginXAngst") ? GetColumn("rlnOriginXAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : (HasColumn("rlnOriginX") ? GetColumn("rlnOriginX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount]);
            float[] Y = HasColumn("rlnOriginYAngst") ? GetColumn("rlnOriginYAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : (HasColumn("rlnOriginY") ? GetColumn("rlnOriginY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount]);
            float[] Z = HasColumn("rlnOriginZAngst") ? GetColumn("rlnOriginZAngst").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : (HasColumn("rlnOriginZ") ? GetColumn("rlnOriginZ").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount]);

            return Helper.Zip(X, Y, Z);
        }

        public float3[] GetRelionAngles()
        {
            float[] X = HasColumn("rlnAngleRot") ? GetColumn("rlnAngleRot").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Y = HasColumn("rlnAngleTilt") ? GetColumn("rlnAngleTilt").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Z = HasColumn("rlnAnglePsi") ? GetColumn("rlnAnglePsi").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];

            return Helper.Zip(X, Y, Z);
        }

        public string[] GetRelionMicrographNames()
        {
            return GetColumn("rlnMicrographName");
        }

        public ValueTuple<string, int>[] GetRelionParticlePaths()
        {
            return GetColumn("rlnImageName").Select(s =>
            {
                string[] Parts = s.Split('@');
                return new ValueTuple<string, int>(Parts[1], int.Parse(Parts[0]) - 1);
            }).ToArray();
        }

        public CTF[] GetRelionCTF()
        {
            float[] Voltage = HasColumn("rlnVoltage") ? GetColumn("rlnVoltage").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(300f, RowCount);
            float[] DefocusU = HasColumn("rlnDefocusU") ? GetColumn("rlnDefocusU").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] DefocusV = HasColumn("rlnDefocusV") ? GetColumn("rlnDefocusV").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] DefocusAngle = HasColumn("rlnDefocusAngle") ? GetColumn("rlnDefocusAngle").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Cs = HasColumn("rlnSphericalAberration") ? GetColumn("rlnSphericalAberration").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(2.7f, RowCount);
            float[] PhaseShift = HasColumn("rlnPhaseShift") ? GetColumn("rlnPhaseShift").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : new float[RowCount];
            float[] Amplitude = HasColumn("rlnAmplitudeContrast") ? GetColumn("rlnAmplitudeContrast").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(0.07f, RowCount);
            float[] Magnification = HasColumn("rlnMagnification") ? GetColumn("rlnMagnification").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(10000f, RowCount);
            float[] PixelSize = HasColumn("rlnDetectorPixelSize") ? GetColumn("rlnDetectorPixelSize").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(1f, RowCount);
            float[] NormCorrection = HasColumn("rlnNormCorrection") ? GetColumn("rlnNormCorrection").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(1f, RowCount);
            float[] BeamTiltX = HasColumn("rlnBeamTiltX") ? GetColumn("rlnBeamTiltX").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(0f, RowCount);
            float[] BeamTiltY = HasColumn("rlnBeamTiltY") ? GetColumn("rlnBeamTiltY").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(0f, RowCount);
            float[] Bfactor = HasColumn("rlnCtfBfactor") ? GetColumn("rlnCtfBfactor").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(0f, RowCount);
            float[] Scale = HasColumn("rlnCtfScalefactor") ? GetColumn("rlnCtfScalefactor").Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray() : Helper.ArrayOfConstant(1f, RowCount);

            CTF[] Result = new CTF[RowCount];

            for (int r = 0; r < RowCount; r++)
            {
                Result[r] = new CTF
                {
                    PixelSize = (decimal)(PixelSize[r] / Magnification[r] * 10000),
                    Voltage = (decimal)Voltage[r],
                    Amplitude = (decimal)Amplitude[r],
                    PhaseShift = (decimal)PhaseShift[r],
                    Cs = (decimal)Cs[r],
                    DefocusAngle = (decimal)DefocusAngle[r],
                    Defocus = (decimal)((DefocusU[r] + DefocusV[r]) * 0.5e-4f),
                    DefocusDelta = (decimal)((DefocusU[r] - DefocusV[r]) * 0.5e-4f),
                    //Scale = (decimal)(NormCorrection[r]),// * Scale[r]),
                    //Bfactor = (decimal)Bfactor[r],
                    BeamTilt = new float2(BeamTiltX[r], BeamTiltY[r])
                };
            }

            return Result;
        }

        public float[] GetFloat(string name1 = null)
        {
            return (name1 == null ? GetColumn(0) : GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }

        public float[] GetFloatRobust(string name1 = null)
        {
            return (name1 == null ? GetColumn(0) : GetColumn(name1)).Select(v => float.Parse(v.Replace("inf", "Infinity").Replace("-nan", "NaN").Replace("nan", "NaN"), CultureInfo.InvariantCulture)).ToArray();
        }

        public float2[] GetFloat2(string name1 = null, string name2 = null)
        {
            float[] Column1 = (name1 == null ? GetColumn(0) : GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? GetColumn(1) : GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2);
        }

        public float3[] GetFloat3(string name1 = null, string name2 = null, string name3 = null)
        {
            float[] Column1 = (name1 == null ? GetColumn(0) : GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? GetColumn(1) : GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = (name3 == null ? GetColumn(2) : GetColumn(name3)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3);
        }

        public float4[] GetFloat4()
        {
            float[] Column1 = GetColumn(0).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = GetColumn(1).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = GetColumn(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column4 = GetColumn(3).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3, Column4);
        }

        public float5[] GetFloat5()
        {
            float[] Column1 = GetColumn(0).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = GetColumn(1).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = GetColumn(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column4 = GetColumn(3).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column5 = GetColumn(4).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3, Column4, Column5);
        }

        public float[][] GetFloatN(int n = -1)
        {
            if (n < 0)
                n = ColumnCount;

            float[][] Result = new float[RowCount][];
            for (int r = 0; r < RowCount; r++)
                Result[r] = GetRow(r).Take(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Result;
        }

        public static float[] LoadFloat(string path, string name1 = null)
        {
            Star TableIn = new Star(path);
            return (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        }

        public static float2[] LoadFloat2(string path, string name1 = null, string name2 = null)
        {
            Star TableIn = new Star(path);

            float[] Column1 = (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? TableIn.GetColumn(1) : TableIn.GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2);
        }

        public static float3[] LoadFloat3(string path, string name1 = null, string name2 = null, string name3 = null)
        {
            Star TableIn = new Star(path);

            float[] Column1 = (name1 == null ? TableIn.GetColumn(0) : TableIn.GetColumn(name1)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = (name2 == null ? TableIn.GetColumn(1) : TableIn.GetColumn(name2)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = (name3 == null ? TableIn.GetColumn(2) : TableIn.GetColumn(name3)).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3);
        }

        public static float4[] LoadFloat4(string path)
        {
            Star TableIn = new Star(path);
            float[] Column1 = TableIn.GetColumn(0).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = TableIn.GetColumn(1).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = TableIn.GetColumn(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column4 = TableIn.GetColumn(3).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3, Column4);
        }

        public static float5[] LoadFloat5(string path)
        {
            Star TableIn = new Star(path);
            float[] Column1 = TableIn.GetColumn(0).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column2 = TableIn.GetColumn(1).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column3 = TableIn.GetColumn(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column4 = TableIn.GetColumn(3).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
            float[] Column5 = TableIn.GetColumn(4).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Helper.Zip(Column1, Column2, Column3, Column4, Column5);
        }

        public static float[][] LoadFloatN(string path, int n = -1)
        {
            Star TableIn = new Star(path);
            if (n < 0)
                n = TableIn.ColumnCount;

            float[][] Result = new float[TableIn.RowCount][];
            for (int r = 0; r < TableIn.RowCount; r++)
                Result[r] = TableIn.GetRow(r).Take(n).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

            return Result;
        }

        public static Dictionary<string, Star> LoadSplitByValue(string path, string columnName)
        {
            var TableIn = new Star(path);
            int ColumnId = TableIn.GetColumnID(columnName);

            Dictionary<string, List<int>> RowIds = new();
            for (int r = 0; r < TableIn.RowCount; r++)
            {
                string RowValue = TableIn.GetRowValue(r, ColumnId);
                if (!RowIds.ContainsKey(RowValue))
                    RowIds.Add(RowValue, new List<int>());
                RowIds[RowValue].Add(r);
            }

            var TablesOut = RowIds.ToDictionary(kvp => kvp.Key,
                                                kvp => TableIn.CreateSubset(kvp.Value));

            return TablesOut;
        }
    }

    public class StarParameters : Star
    {
        public StarParameters(string[] parameterNames, string[] parameterValues) :
            base(parameterValues.Select(v => new string[] { v }).ToArray(), parameterNames)
        {
        }

        public StarParameters(string path, string tableName = "", int nrows = -1)
        {
            using (TextReader Reader = new StreamReader(File.OpenRead(path)))
            {
                string Line;

                if (!string.IsNullOrEmpty(tableName))
                {
                    tableName = "data_" + tableName;
                    while((Line = Reader.ReadLine()) != null && !Line.StartsWith(tableName)) ;
                }

                while((Line = Reader.ReadLine()) != null && !Line.StartsWith("_")) ;

                List<string> OnlyRow = new List<string>();

                while(true)
                {
                    if (Line == null)
                        break;
                    if (Line[0] == '_')
                    {
                        string[] Parts = Line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        if (Parts.Length < 2)
                            throw new Exception("STAR file is poorly formatted, expected at least a column name and a value in this line: " + Line);

                        string ColumnName = Parts[0].Substring(1);
                        string ColumnValue = Parts[1];
                        int ColumnIndex = NameMapping.Count;
                        NameMapping.Add(ColumnName, ColumnIndex);

                        OnlyRow.Add(ColumnValue);
                    }

                    Line = Reader.ReadLine();
                    if (Line == null || !Line.StartsWith("_"))
                        break;
                }

                Rows.Add(OnlyRow.ToArray());
            }
        }

        public override void Save(string path, string name = "", bool append = false)
        {
            using (TextWriter Writer = append ? File.AppendText(path) : File.CreateText(path))
            {
                if (append)
                    Writer.Write("\n\n\n");

                Writer.Write("\n");
                Writer.Write("data_" + name + "\n");
                Writer.Write("\n");

                int[] ColumnWidths = new int[2];

                foreach (var pair in NameMapping)
                    ColumnWidths[0] = Math.Max(ColumnWidths[0], pair.Key.Length + 1);
                foreach (var value in Rows[0])
                    ColumnWidths[1] = Math.Max(ColumnWidths[1], value.Length);

                foreach (var pair in NameMapping)
                {
                    StringBuilder NameBuilder = new StringBuilder();
                    NameBuilder.Append("_" + pair.Key);
                    NameBuilder.Append(' ', ColumnWidths[0] - pair.Key.Length);

                    StringBuilder ValueBuilder = new StringBuilder();
                    ValueBuilder.Append(' ', ColumnWidths[1] - Rows[0][pair.Value].Length);
                    ValueBuilder.Append(Rows[0][pair.Value]);

                    Writer.Write(NameBuilder + "  " + ValueBuilder + "\n");
                }
            }
        }
    }

    public class SpanLineReader : IDisposable
    {
        private const int DefaultBufferSize = 1 << 16; // 64 KB should be enough for anybody
        private readonly Stream Stream;
        private byte[] Buffer;
        private int BufferEnd = 0;
        private int BufferStart = 0;
        private char[] LineBuffer;

        public SpanLineReader(string path, int bufferSize = DefaultBufferSize)
        {
            Stream = File.OpenRead(path);
            Buffer = new byte[bufferSize];
            LineBuffer = new char[bufferSize];
        }

        public void Dispose()
        {
            Stream?.Dispose();
        }

        public string? ReadLine()
        {
            ReadOnlySpan<char> LineSpan;
            bool Success = TryReadLineSpan(out LineSpan);
            if (Success)
                return LineSpan.ToString();
            else
                return null;
        }

        public bool TryReadLineSpan(out ReadOnlySpan<char> line)
        {
            while(true)
            {
                int newLineIndex = Array.IndexOf(Buffer, (byte)'\n', BufferStart, BufferEnd - BufferStart);
                if (newLineIndex >= 0)
                {
                    int lineStart = BufferStart;
                    int lineEnd = newLineIndex - ((newLineIndex > 0 && Buffer[newLineIndex - 1] == '\r') ? 1 : 0);

                    BufferStart = newLineIndex + 1;

                    int NChars = Encoding.UTF8.GetCharCount(Buffer, lineStart, lineEnd - lineStart);
                    if (LineBuffer.Length < NChars)
                        LineBuffer = new char[NChars];

                    Encoding.UTF8.GetChars(new ReadOnlySpan<byte>(Buffer).Slice(lineStart, lineEnd - lineStart), LineBuffer.AsSpan());
                    line = new ReadOnlySpan<char>(LineBuffer).Slice(0, NChars);

                    return true;
                }

                // If we get to this point, it means there are no '\n' in the buffer, and we should try to refill.
                // But before refilling, let's check if the buffer is full. If it is, we need to increase its size.
                if (BufferStart == 0 && BufferEnd == Buffer.Length)
                {
                    ExpandBuffer();
                }

                if (!RefillBuffer())
                {
                    if (BufferStart != BufferEnd)
                    {
                        int NChars = Encoding.UTF8.GetCharCount(Buffer, BufferStart, BufferEnd - BufferStart);
                        if (LineBuffer.Length < NChars)
                            LineBuffer = new char[NChars];

                        Encoding.UTF8.GetChars(new ReadOnlySpan<byte>(Buffer).Slice(BufferStart, BufferEnd - BufferStart), LineBuffer.AsSpan());
                        line = new ReadOnlySpan<char>(LineBuffer).Slice(0, NChars);

                        BufferStart = BufferEnd;
                        return true;
                    }

                    line = default;
                    return false;
                }
            }
        }

        private void ExpandBuffer()
        {
            var newBuffer = new byte[Buffer.Length * 2];
            Array.Copy(Buffer, newBuffer, Buffer.Length);
            Buffer = newBuffer;
        }

        private bool RefillBuffer()
        {
            if (BufferStart > 0)
            {
                BufferEnd -= BufferStart;
                Array.Copy(Buffer, BufferStart, Buffer, 0, BufferEnd);
                BufferStart = 0;
            }

            int readBytes = Stream.Read(Buffer, BufferEnd, Buffer.Length - BufferEnd);
            if (readBytes > 0)
            {
                BufferEnd += readBytes;
                return true;
            }

            return false;
        }
    }
}