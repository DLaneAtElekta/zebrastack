using CsvHelper.Configuration.Attributes;

namespace HerringstackApi.Services;

internal class CsvBoundingBoxMetadata
{
    [Name("Image Index")]
    public string Id { get; set; }

    [Name("Finding Label")]
    public string FindingLabel { get; set; }

    [Name("Bbox [x")]
    public decimal BoundingBoxX { get; set; }

    [Name("y")]
    public decimal BoundingBoxY { get; set; }

    [Name("w")]
    public decimal BoundingBoxWidth { get; set; }

    [Name("h]")]
    public decimal BoundingBoxHeight { get; set; }
}