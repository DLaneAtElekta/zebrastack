namespace HerringstackApi.Abstractions;

public class BoundingBox
{
    public string ImageKey { get; set; }

    public int SubjectID { get; set; }

    public string FindingLabel { get; set; }

    public decimal[] UpperLeft { get; set; }

    public decimal[] Extent { get; set; }
}
