from pydantic import BaseModel, Field
from typing import List


class ImageOCRParagraph(BaseModel):
    text: str = Field(..., title="Paragraph Text",
                      description="The text content of the paragraph. Should not include hyphens indicating word breaks at line/page endings. Broken words within paragraphs should be merged.")
    tag: str | None = Field(..., title="Paragraph Tag",
                            description="An optional tag indicating the type of paragraph per custom user-provided instructions")


class ImageOCRPage(BaseModel):
    page_number: int | None = Field(..., title="Page Number",
                                    description="The page number of the image, if visible or can be deduced")
    chapter_title: str | None = Field(..., title="Chapter Title",
                                      description="The chapter title, if visible")
    paragraphs: List[ImageOCRParagraph] = Field(..., title="Paragraphs",
                                                description="List of paragraphs extracted from the page")


class ImageOCRResult(BaseModel):
    pages: List[ImageOCRPage] = Field(..., title="Pages",
                                      description="List of pages processed (usually one or two per image)")


class ChapterContent(BaseModel):
    title: str = Field(..., title="Chapter Title", description="The chapter title")
    paragraphs: List[ImageOCRParagraph] = Field(..., title="Paragraphs",
                                                description="All paragraphs in the chapter")


class BookMergeResult(BaseModel):
    chapters: List[ChapterContent] = Field(..., title="Chapters",
                                           description="List of chapters with merged content")

