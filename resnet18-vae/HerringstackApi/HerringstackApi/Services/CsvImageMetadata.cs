namespace HerringstackApi.Services;
using CsvHelper.Configuration.Attributes;

internal class CsvImageMetadata
{
    [Name("Image Index")]
    public string Id { get; set; }

    [Name("Finding Labels")]
    public string FindingLabels { get; set; }

    [Name("Follow-up #")]
    public int FollowupNumber { get; set; }

    [Name("Patient ID")]
    public int PatientID { get; set; }

    [Name("Patient Age")]
    public int PatientAge { get; set; }

    [Name("Patient Gender")]
    public string PatientGender { get; set; }

    [Name("View Position")]
    public string ViewPosition { get; set; }

    [Name("OriginalImage[Width")]
    public int OriginalImageWidth { get; set; }

    [Name("Height]")]
    public int OriginalImageHeight { get; set; }

    [Name("OriginalImagePixelSpacing[x")]
    public decimal OriginalImagePixelSpacingX { get; set; }

    [Name("y]")]
    public decimal OriginalImagePixelSpacingY { get; set; }
}
