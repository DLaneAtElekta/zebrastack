using HerringstackApi.Abstractions;
using HerringstackApi.Services;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace HerringstackApi.Controllers;

[Route("api/[controller]")]
[ApiController]
public class SubjectsController : ControllerBase
{
    private readonly ICxr8DataManager _dataManager;

    public SubjectsController(ICxr8DataManager dataManager)
    {
        _dataManager = dataManager;
    }

    // GET: api/<SubjectsController>
    [HttpGet]
    [Produces("application/json", Type = typeof(IEnumerable<Subject>))]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<IActionResult> GetSubjects([FromQuery] int pageSize, [FromQuery] int pageNumber)
    {
        IEnumerable<Subject> subjectItems = await _dataManager.GetSubjectItemsAsync(pageSize, pageNumber);
        return Ok(subjectItems);
    }

    // GET: api/<SubjectsController>/5
    [HttpGet("{id}")]
    [Produces("application/json", Type = typeof(Subject))]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> GetSubjectById(int id)
    {
        IEnumerable<Subject> subjectItems = 
            (await _dataManager.GetSubjectItemsAsync())
            .Where(subject => subject.SubjectID == id);
        if (!subjectItems.Any()) 
        { 
            return NotFound();
        }

        if (subjectItems.Count() > 1)
        {
            return StatusCode(StatusCodes.Status500InternalServerError);
        }

        Subject subjectItem = subjectItems.Where(subject => subject.SubjectID == id).Single();
        return Ok(subjectItem);
    }

    // GET api/<SubjectsController>/5/images
    [HttpGet("{id}/images")]
    [Produces("application/json", Type = typeof(IEnumerable<Image>))]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<IActionResult> GetImagesForSubject(int id)
    {
        IEnumerable<Image> images = await _dataManager.GetImagesForSubjectAsync(id);
        return Ok(images);
    }

    // GET api/<SubjectController>/5/images/00000005_002/pixels
    [HttpGet("{id}/images/{imagekey}/pixels")]
    [Produces("image/png")]
    [ProducesResponseType(200, Type = typeof(Stream))]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> GetImagePixels(int id, string imagekey,
        [FromQuery] string format = "png", [FromQuery] string processed = "none")
    {
        var pixelBytes = await _dataManager.GetImagePixelsAsync(id, imagekey, format: format, processed: processed);
        if (pixelBytes.IsFailed)
        {
            return NotFound(pixelBytes);
        }

        return File(pixelBytes.Value, $"image/{format}");
    }

    // GET api/<SubjectController>/5/images/00000005_002/boundingboxes
    [HttpGet("{id}/images/{imagekey}/boundingboxes")]
    [Produces("application/json", Type = typeof(IEnumerable<BoundingBox>))]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<IEnumerable<BoundingBox>>> GetBoundingBoxes(int id, string imagekey)
    {
        IEnumerable<BoundingBox> boundingBoxes = await _dataManager.GetBoundingBoxesForImageAsync(id, imagekey);
        //if (!boundingBoxes.Any())
        //{
        //    return NotFound();
        //}

        return Ok(boundingBoxes);
    }
}
