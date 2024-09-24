using System.Text;

namespace Bridge.Components.Verbs
{
    public class VerbCommand
    {
        public string Name;
        public List<string> Arguments = new();

        public VerbCommand(string name)
        {
            Name = name;
        }

        public void AddArgument(string arg)
        {
            Arguments.Add(arg);
        }

        public string GetString(bool multiline)
        {
            if (Arguments.Count == 0)
                return Name;

            StringBuilder sb = new();
            sb.Append(Name);

            for (int i = 0; i < Arguments.Count; i++)
            {
                if (multiline)
                {
                    if (i == 0)
                        sb.Append(" \\\n");

                    if (i < Arguments.Count - 1)
                        sb.Append(Arguments[i] + " \\\n");
                    else
                        sb.Append(Arguments[i]);
                }
                else
                    sb.Append(" " + Arguments[i]);
            }

            return sb.ToString();
        }
    }
}
