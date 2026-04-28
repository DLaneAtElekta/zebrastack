namespace HerringstackApi.Abstractions;

public class Image
{
    public string Key { get; set; }

    public int PatientID { get; set; }

    public int PatientAge { get; set; }

    public int FollowupNumber { get; set; }

    public string ViewPosition { get; set; }

    public string[] FindingLabels { get; set; }
}