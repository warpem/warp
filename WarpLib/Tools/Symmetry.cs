using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class Symmetry
    {
        public SymmetryTypes Type = SymmetryTypes.Cn;
        public int Multiplicity = 1;

        public Symmetry()
        {
        }

        public Symmetry(SymmetryTypes type, int multiplicity)
        {
            Type = type;
            Multiplicity = multiplicity;
        }

        public Symmetry(string symString)
        {
            Type = ParseType(symString);

            if (Type == SymmetryTypes.CI ||
                Type == SymmetryTypes.CS ||
                Type == SymmetryTypes.T ||
                Type == SymmetryTypes.TD ||
                Type == SymmetryTypes.TH ||
                Type == SymmetryTypes.O ||
                Type == SymmetryTypes.OH ||
                Type == SymmetryTypes.I ||
                Type == SymmetryTypes.IH)
            {
                Multiplicity = 1;
            }
            else
            {
                // Sanitize
                while (symString.Length < 4)
                    symString += " ";

                if (char.IsDigit(symString[1]) && char.IsDigit(symString[2]))
                    Multiplicity = int.Parse(symString.Substring(1, 2));
                else
                    Multiplicity = int.Parse(symString.Substring(1, 1));

                Multiplicity = Math.Max(1, Math.Min(Multiplicity, 99));
            }
        }

        public Matrix3[] GetRotationMatrices()
        {
            int NMatrices = CPU.SymmetryGetNumberOfMatrices(this.ToString());
            float[] Values = new float[NMatrices * 9];

            CPU.SymmetryGetMatrices(this.ToString(), Values);

            Matrix3[] Result = new Matrix3[NMatrices];
            for (int i = 0; i < NMatrices; i++)
                Result[i] = new Matrix3(Values.Skip(i * 9).Take(9).ToArray()).Transposed();

            return Result;
        }

        public override string ToString()
        {
            switch (Type)
            {
                case SymmetryTypes.Cn:
                    return $"C{Multiplicity}";
                case SymmetryTypes.CI:
                    return "CI";
                case SymmetryTypes.CS:
                    return "CS";
                case SymmetryTypes.CnH:
                    return $"C{Multiplicity}H";
                case SymmetryTypes.CnV:
                    return $"C{Multiplicity}V";
                case SymmetryTypes.Sn:
                    return $"S{Multiplicity}";
                case SymmetryTypes.Dn:
                    return $"D{Multiplicity}";
                case SymmetryTypes.DnH:
                    return $"D{Multiplicity}H";
                case SymmetryTypes.DnV:
                    return $"D{Multiplicity}V";
                case SymmetryTypes.T:
                    return "T";
                case SymmetryTypes.TD:
                    return "TD";
                case SymmetryTypes.TH:
                    return "TH";
                case SymmetryTypes.O:
                    return "O";
                case SymmetryTypes.OH:
                    return "OH";
                case SymmetryTypes.I:
                    return "I";
                case SymmetryTypes.In:
                    return $"I{Multiplicity}";
                case SymmetryTypes.IH:
                    return "IH";
                case SymmetryTypes.InH:
                    return $"I{Multiplicity}H";
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static SymmetryTypes ParseType(string symString)
        {
            if (symString.Length < 1 || symString.Length > 4)
                throw new ArgumentOutOfRangeException();

            // Sanitize
            symString = symString.ToUpper();

            if (symString == "CI")
                return SymmetryTypes.CI;
            if (symString == "CS")
                return SymmetryTypes.CS;
            if (symString == "T")
                return SymmetryTypes.T;
            if (symString == "TD")
                return SymmetryTypes.TD;
            if (symString == "TH")
                return SymmetryTypes.TH;
            if (symString == "O")
                return SymmetryTypes.O;
            if (symString == "OH")
                return SymmetryTypes.OH;
            if (symString == "I")
                return SymmetryTypes.I;
            if (symString == "IH")
                return SymmetryTypes.IH;

            if (symString == "CN")
                return SymmetryTypes.Cn;
            if (symString == "CNH")
                return SymmetryTypes.CnH;
            if (symString == "CNV")
                return SymmetryTypes.CnV;
            if (symString == "SN")
                return SymmetryTypes.Sn;
            if (symString == "DN")
                return SymmetryTypes.Dn;
            if (symString == "DNH")
                return SymmetryTypes.DnH;
            if (symString == "DNV")
                return SymmetryTypes.DnV;
            if (symString == "IN")
                return SymmetryTypes.In;
            if (symString == "INH")
                return SymmetryTypes.InH;

            // Sanitize
            while (symString.Length < 4)
                symString += " ";

            if (symString[0] == 'C')
            {
                if (char.IsDigit(symString[1]))
                {
                    if (symString[2] == 'H')
                        return SymmetryTypes.CnH;
                    if (symString[2] == 'V')
                        return SymmetryTypes.CnV;
                    if (char.IsDigit(symString[2]))
                    {
                        if (symString[3] == 'H')
                            return SymmetryTypes.CnH;
                        if (symString[3] == 'V')
                            return SymmetryTypes.CnV;
                    }

                    return SymmetryTypes.Cn;
                }
            }

            if (symString[0] == 'S' && char.IsDigit(symString[1]))
                return SymmetryTypes.Sn;

            if (symString[0] == 'D')
            {
                if (char.IsDigit(symString[1]))
                {
                    if (symString[2] == 'H')
                        return SymmetryTypes.DnH;
                    if (symString[2] == 'V')
                        return SymmetryTypes.DnV;
                    if (char.IsDigit(symString[2]))
                    {
                        if (symString[3] == 'H')
                            return SymmetryTypes.DnH;
                        if (symString[3] == 'V')
                            return SymmetryTypes.DnV;
                    }

                    return SymmetryTypes.Dn;
                }
            }

            if (symString[0] == 'I')
            {
                if (char.IsDigit(symString[1]))
                {
                    if (symString[2] == 'H')
                        return SymmetryTypes.InH;
                    if (char.IsDigit(symString[2]))
                    {
                        if (symString[3] == 'H')
                            return SymmetryTypes.InH;
                    }

                    return SymmetryTypes.In;
                }
            }

            throw new Exception("Unknown symmetry");
        }

        public static bool HasMultiplicity(SymmetryTypes symType)
        {
            return !(symType == SymmetryTypes.CI ||
                     symType == SymmetryTypes.CS ||
                     symType == SymmetryTypes.T ||
                     symType == SymmetryTypes.TD ||
                     symType == SymmetryTypes.TH ||
                     symType == SymmetryTypes.O ||
                     symType == SymmetryTypes.OH ||
                     symType == SymmetryTypes.I ||
                     symType == SymmetryTypes.IH);
        }
    }

    public enum SymmetryTypes
    {
            Cn,
            CI,
            CS,
            CnH,
            CnV,
            Sn,
            Dn,
            DnH,
            DnV,
            T,
            TD,
            TH,
            O,
            OH,
            I,
            In,
            IH,
            InH
    }

    public class HelicalSymmetry
    {
        public int Units;
        public float Twist;
        public float Rise;

        public HelicalSymmetry(int units, float twist, float rise)
        {
            Units = units;
            Twist = twist;
            Rise = rise;
        }
    }
}
