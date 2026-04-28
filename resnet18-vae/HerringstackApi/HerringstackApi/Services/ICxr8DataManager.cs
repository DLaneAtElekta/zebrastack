namespace HerringstackApi.Services;

using FluentResults;
using HerringstackApi.Abstractions;
using System.Collections.Generic;
using System.Threading.Tasks;

/// <summary>
/// 
/// </summary>
public interface ICxr8DataManager
{
    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    Task<IList<Subject>> GetSubjectItemsAsync(int pageSize = 0, int pageNumber = 1);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <returns></returns>
    Task<IList<Image>> GetImagesForSubjectAsync(int subjectId);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <param name="imageKey"></param>
    /// <returns></returns>
    Task<IList<BoundingBox>> GetBoundingBoxesForImageAsync(int subjectId, string imageKey);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectId"></param>
    /// <param name="imageKey"></param>
    /// <returns></returns>
    Task<IList<ImageLatentVector>> GetLatentVectorsForImageAsync(int subjectId, string imageKey);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="subjectid"></param>
    /// <param name="imagekey"></param>
    /// <param name="processed"></param>
    /// <param name="format"></param>
    /// <returns></returns>
    Task<Result<byte[]>> GetImagePixelsAsync(int subjectid, string imagekey, string processed = "none", string format = "png");
}
