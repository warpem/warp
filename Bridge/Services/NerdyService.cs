namespace Bridge.Services
{
    public class NerdyService
    {
        private bool _isNerdy;

        public bool IsNerdy
        {
            get => _isNerdy;
            set
            {
                if (_isNerdy != value)
                {
                    _isNerdy = value;
                    OnIsNerdyChanged?.Invoke(_isNerdy);
                }
            }
        }

        public event Action<bool> OnIsNerdyChanged;
    }
}
