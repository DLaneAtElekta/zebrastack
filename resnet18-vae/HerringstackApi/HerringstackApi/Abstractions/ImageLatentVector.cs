namespace HerringstackApi.Abstractions;

public class ImageLatentVector
{
    public string ModelName { get; set; }

    public string ModelVersion { get; set; }

    public string ImageKey { get; set; }

    public decimal[] LatentVector { get; set; }
}