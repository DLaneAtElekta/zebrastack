using HerringstackApi.Services;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<ICxr8DataManager, Cxr8DataManager>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseStaticFiles(new StaticFileOptions
{
    // Trace.Assert(app.Configuration.Get)
    FileProvider = new PhysicalFileProvider("C:\\data\\cxr8\\"), // app.Configuration["DataBasePath"]),
    RequestPath = "/DataBaseRoot",
    ServeUnknownFileTypes = true
});

app.UseAuthorization();

app.MapControllers();

app.Run();
