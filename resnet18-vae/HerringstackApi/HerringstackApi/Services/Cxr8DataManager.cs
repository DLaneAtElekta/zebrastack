namespace HerringstackApi.Services;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CsvHelper;
using HerringstackApi.Abstractions;
using Microsoft.Extensions.Configuration;
using FluentResults;
using TorchSharp;

/// <summary>
/// 
/// </summary>
public class Cxr8DataManager : ICxr8DataManager
{
    private readonly string _basePath;
    private readonly string _csvMetadataPath;
    private readonly string _csvBoundingBoxPath;

    private readonly string _imagePath;
    private readonly string _clahePath;
    private readonly string _reconPath;

    private readonly torch.nn.Module _model;

    private IList<CsvImageMetadata> _metadataItems;
    private IList<CsvBoundingBoxMetadata> _boundingBoxItems;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="configuration"></param>
    public Cxr8DataManager(IConfiguration configuration)
    {
        _basePath = configuration["DataBasePath"];

        _csvMetadataPath = configuration["MetadataCsvName"];
        _csvBoundingBoxPath = configuration["BoundingBoxCsvName"];
        _imagePath = configuration["ImageSubpath"];
        _clahePath = configuration["ClaheSubpath"];
        _reconPath = configuration["ReconSubpath"];

        var modelPath = configuration["ModelPath"];
        var state_dict = torch.load(modelPath);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    public async Task<IList<Subject>> GetSubjectItemsAsync(int pageSize, int pageNumber)
    {
        var metadataItems = await GetMetadataItemsAsync();
        IEnumerable<IGrouping<int, CsvImageMetadata>> patientGroups =
            metadataItems.GroupBy(imd => imd.PatientID)
                .OrderBy(grp => grp.Key);

        if (pageSize > 0)
        {
            patientGroups = patientGroups.Skip(pageSize * (pageNumber - 1)).Take(pageSize);
        }
        
        // TODO: use AutoMapper for this
        var subjectMetadatas =
            patientGroups.Select(grp =>
                new Subject()
                {
                    SubjectID = grp.Key,
                    SubjectGender = grp.First().PatientGender,
                    SubjectAgeRange = GetSubjectAgeRange(grp),
                    ImageFindings = GetImageFindings(grp),
                    ImageCount = grp.Count(),
                });

        return subjectMetadatas.ToList();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <returns></returns>
    public async Task<IList<Image>> GetImagesForSubjectAsync(int subjectId)
    {
        var metadataItems = await GetMetadataItemsAsync();
        var imagesForSubject = metadataItems.Where(imd => imd.PatientID == subjectId);

        // TODO: use AutoMapper to convert
        var converted =
            imagesForSubject.Select(imd => new Image()
            {
                Key = imd.Id.Split(".").First(),
                PatientID = imd.PatientID,
                FindingLabels = imd.FindingLabels.Split('|'),
                FollowupNumber = imd.FollowupNumber,
                PatientAge = imd.PatientAge,
                ViewPosition = imd.ViewPosition,
            });

        return converted.ToList();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <param name="imageKey"></param>
    /// <returns></returns>
    public async Task<IList<BoundingBox>> GetBoundingBoxesForImageAsync(int subjectId, string imageKey)
    {
        var boundingBoxItems = await GetBoundingBoxItemsAsync();
        var boundingBoxesForImage = boundingBoxItems.Where(bbi => bbi.Id.CompareTo($"{imageKey}.png") == 0);

        // TODO: use AutoMapper to convert
        var converted =
            boundingBoxesForImage.Select(bbi => new BoundingBox()
            {
                ImageKey = bbi.Id.Split(".").First(),
                SubjectID = subjectId,
                FindingLabel = bbi.FindingLabel,
                UpperLeft = [bbi.BoundingBoxX, bbi.BoundingBoxY],
                Extent = [bbi.BoundingBoxWidth, bbi.BoundingBoxHeight],
            });

        return converted.ToList();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <param name="imageKey"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public Task<IList<ImageLatentVector>> GetLatentVectorsForImageAsync(int subjectId, string imageKey)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectid"></param>
    /// <param name="imagekey"></param>
    /// <param name="processed"></param>
    /// <param name="format"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public async Task<Result<byte[]>> GetImagePixelsAsync(int subjectid, string imagekey, string processed = "none", string format = "png")
    {
        processed = processed.ToLower();
        var path = $"{_basePath}/{_imagePath}/{imagekey}.{format}";
        if (!System.IO.File.Exists(path))
        {
            return Result.Fail("Not found");
        }

        byte[] pixelBytes = await File.ReadAllBytesAsync(path);
        if (processed == "recon")
        {
            var unprocessPixels = await GetImagePixelsAsync(subjectid, imagekey, processed = "none", format = "png");
            if (unprocessPixels.IsFailed)
            {
                return unprocessPixels;
            }

            // TODO: load and call model

            pixelBytes = unprocessPixels.Value;
        }
        else if (processed != "none")
        {
            return Result.Fail($"Invalid string value for processed = {processed}");
        }

        return Result.Ok(pixelBytes);
    }

    async Task<IList<CsvImageMetadata>> GetMetadataItemsAsync()
    {
        if (_metadataItems == null)
        {
            _metadataItems = await GetCsvItemsAsync<CsvImageMetadata>(_csvMetadataPath);
        }

        return _metadataItems;
    }

    async Task<IList<CsvBoundingBoxMetadata>> GetBoundingBoxItemsAsync()
    {
        if (_boundingBoxItems == null)
        {
            _boundingBoxItems = await GetCsvItemsAsync<CsvBoundingBoxMetadata>(_csvBoundingBoxPath);
        }

        return _boundingBoxItems.ToList();
    }

    async Task<IList<T>> GetCsvItemsAsync<T>(string path)
    {
        using var reader = new StreamReader($"{_basePath}\\{path}");
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        return csv.GetRecords<T>().ToList();
    }

    Tuple<int, int> GetSubjectAgeRange(IEnumerable<CsvImageMetadata> forMetadatas)
    {
        var sorted = forMetadatas.OrderBy(imd => imd.PatientAge);
        return Tuple.Create(sorted.First().PatientAge, sorted.Last().PatientAge);
    }

    IList<string> GetImageFindings(IEnumerable<CsvImageMetadata> forMetadatas) =>
        forMetadatas.SelectMany(imd => imd.FindingLabels.Split('|')).Distinct().ToList();
}
