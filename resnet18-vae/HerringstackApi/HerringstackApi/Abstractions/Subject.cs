namespace HerringstackApi.Abstractions;

public class Subject
{
    public int SubjectID { get; set; }

    public string SubjectGender { get; set; }

    public Tuple<int, int> SubjectAgeRange { get; set; }

    public int ImageCount { get; set; }

    public IList<string> ImageFindings { get; set; }
}
