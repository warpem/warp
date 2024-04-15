using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace M
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            base.OnStartup(e);

            if (!Debugger.IsAttached)
            {
                AppDomain.CurrentDomain.UnhandledException += CurrentDomainOnUnhandledException;
                Application.Current.DispatcherUnhandledException += DispatcherOnUnhandledException;
                TaskScheduler.UnobservedTaskException += TaskSchedulerOnUnobservedTaskException;
            }
        }

        private void TaskSchedulerOnUnobservedTaskException(object sender, UnobservedTaskExceptionEventArgs unobservedTaskExceptionEventArgs)
        {
            SendReport(unobservedTaskExceptionEventArgs.Exception);
            Environment.Exit(0);
        }

        private void DispatcherOnUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs dispatcherUnhandledExceptionEventArgs)
        {
            SendReport(dispatcherUnhandledExceptionEventArgs.Exception);
            Environment.Exit(0);
        }

        private static void CurrentDomainOnUnhandledException(object sender, UnhandledExceptionEventArgs unhandledExceptionEventArgs)
        {
            SendReport((Exception)unhandledExceptionEventArgs.ExceptionObject);
            Environment.Exit(0);
        }

        public static void SendReport(Exception exception)
        {
            using (TextWriter writer = File.CreateText("lastcrash.txt"))
            {
                writer.WriteLine(DateTime.Now.ToString() + ":");
                writer.WriteLine(exception.ToString());
                writer.WriteLine(new StackTrace(exception, true).ToString());
            }

            var Result = MessageBox.Show("Damn, M just crashed!\n" +
                                         $"Here are the details that were also saved in {Environment.CurrentDirectory}\\lastcrash.txt.\n" +
                                         "\n" +
                                         exception.ToString() +
                                         "\n" +
                                         new StackTrace(exception, true).ToString() +
                                         "\n" +
                                         "Please consider reporting the issue in https://groups.google.com/forum/#!forum/warp-em.\n" +
                                         "Would you like to be taken there now?",
                                         "OH NOES!",
                                         MessageBoxButton.YesNo,
                                         MessageBoxImage.Error);

            if (Result == MessageBoxResult.Yes)
                Process.Start("https://groups.google.com/forum/#!forum/warp-em");
        }
    }
}
