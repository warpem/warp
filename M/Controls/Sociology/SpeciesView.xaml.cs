using M.Controls.Sociology.Dialogs;
using MahApps.Metro.Controls.Dialogs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace M.Controls.Sociology
{
    /// <summary>
    /// Interaction logic for SpeciesView.xaml
    /// </summary>
    public partial class SpeciesView : UserControl
    {
        public Species Species
        {
            get { return (Species)GetValue(SpeciesProperty); }
            set { SetValue(SpeciesProperty, value); }
        }
        public static readonly DependencyProperty SpeciesProperty = DependencyProperty.Register("Species", typeof(Species), typeof(SpeciesView), new PropertyMetadata(null, (sender, args) => ((SpeciesView)sender).DataContext = args.NewValue));
        

        public SpeciesView()
        {
            InitializeComponent();
        }

        private void ButtonParticles_Click(object sender, RoutedEventArgs e)
        {
            MenuParticles.IsOpen = true;
        }

        private async void AddParticles_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogSpeciesParticleSets NewDialogContent = new DialogSpeciesParticleSets(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);

            NewDialogContent.Add += async () =>
            {
                await Window.HideMetroDialogAsync(NewDialog);

                var NewSpeciesProgress = await Window.ShowProgressAsync("Please wait while particle statistics are updated...",
                                                                        "");
                NewSpeciesProgress.SetIndeterminate();
                
                await Task.Run(() =>
                {
                    S.ReplaceParticles(NewDialogContent.ParticlesFinal);

                    S.CalculateParticleStats();

                    S.Commit();
                    S.Save();
                });

                // Make sure all displays are updated
                Species = null;
                Species = S;

                await NewSpeciesProgress.CloseAsync();
            };

            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }

        private async void ExpandSym_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogSpeciesSymExpand NewDialogContent = new DialogSpeciesSymExpand(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);

            NewDialogContent.Expand += async () =>
            {
                S.ReplaceParticles(NewDialogContent.ParticlesFinal);
                S.Symmetry = NewDialogContent.SpeciesSymmetryRemaining;

                await Window.HideMetroDialogAsync(NewDialog);

                var NewSpeciesProgress = await Window.ShowProgressAsync("Please wait while particle statistics are updated...",
                                                                        "");
                NewSpeciesProgress.SetIndeterminate();

                await Task.Run(() =>
                {
                    S.CalculateParticleStats();

                    S.Commit();
                    S.Save();
                });

                // Make sure all displays are updated
                Species = null;
                Species = S;

                await NewSpeciesProgress.CloseAsync();
            };

            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }

        private async void RotateParticles_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogSpeciesRotate NewDialogContent = new DialogSpeciesRotate(Window.ActivePopulation, S, S.MapDenoised);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);

            NewDialogContent.Rotate += async () =>
            {
                S.ReplaceParticles(NewDialogContent.ParticlesFinal);

                var NewSpeciesProgress = await Window.ShowProgressAsync("Please wait while the maps are being rotated...",
                                                                        "");
                NewSpeciesProgress.SetIndeterminate();

                await Task.Run(() =>
                {
                    GPU.SetDevice(GPU.GetDeviceWithMostMemory());

                    float3 Angle = new float3(0);

                    Dispatcher.Invoke(() => Angle = new float3((float)NewDialogContent.AngleRot,
                                                               (float)NewDialogContent.AngleTilt,
                                                               (float)NewDialogContent.AnglePsi) * Helper.ToRad);

                    S.MapFiltered = S.MapFiltered.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.MapFilteredSharpened = S.MapFilteredSharpened.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.MapFilteredAnisotropic = S.MapFilteredAnisotropic.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.MapLocallyFiltered = S.MapLocallyFiltered.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.MapDenoised = S.MapDenoised.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.HalfMap1 = S.HalfMap1.AsRotated3D(Angle).FreeDevice().AndDisposeParent();
                    S.HalfMap2 = S.HalfMap2.AsRotated3D(Angle).FreeDevice().AndDisposeParent();

                    S.Mask = S.Mask.AsRotated3D(Angle).AndDisposeParent();
                    S.Mask.Binarize(0.8f);
                    S.Mask.FreeDevice();

                    S.Commit();
                    S.Save();
                });

                await Window.HideMetroDialogAsync(NewDialog);

                // Make sure all displays are updated
                Species = null;
                Species = S;

                await NewSpeciesProgress.CloseAsync();
            };

            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }

        private async void ShiftParticles_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogSpeciesShift NewDialogContent = new DialogSpeciesShift(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);

            NewDialogContent.Shift += async () =>
            {
                S.ReplaceParticles(NewDialogContent.ParticlesFinal);

                var NewSpeciesProgress = await Window.ShowProgressAsync("Please wait while the maps are being shifted...",
                                                                        "");
                NewSpeciesProgress.SetIndeterminate();

                await Task.Run(() =>
                {
                    float3 Shift = new float3(0);
                    
                    Dispatcher.Invoke(() => Shift = -new float3((float)NewDialogContent.ShiftX, 
                                                                (float)NewDialogContent.ShiftY, 
                                                                (float)NewDialogContent.ShiftZ) / (float)S.PixelSize);

                    S.MapFiltered = S.MapFiltered.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.MapFilteredSharpened = S.MapFilteredSharpened.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.MapFilteredAnisotropic = S.MapFilteredAnisotropic.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.MapLocallyFiltered = S.MapLocallyFiltered.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.MapDenoised = S.MapDenoised.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.HalfMap1 = S.HalfMap1.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();
                    S.HalfMap2 = S.HalfMap2.AsShiftedVolume(Shift).FreeDevice().AndDisposeParent();

                    S.Mask = S.Mask.AsShiftedVolume(Shift).AndDisposeParent();
                    S.Mask.Binarize(0.5f);
                    S.Mask.FreeDevice();

                    S.Commit();
                    S.Save();
                });

                await Window.HideMetroDialogAsync(NewDialog);

                // Make sure all displays are updated
                Species = null;
                Species = S;

                await NewSpeciesProgress.CloseAsync();
            };

            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }

        private async void ExportSubtomo_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoParticleExport NewDialogContent = new DialogTomoParticleExport(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);
            
            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }
    }
}
