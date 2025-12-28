from pydantic import BaseModel, Field
from typing import List


class ImageOCRParagraph(BaseModel):
    text: str = Field(..., title="Paragraph Text",
                      description="The text content of the paragraph. Should not include hyphens indicating word breaks at line/page endings. Broken words within the paragraph should be merged. The last paragraph on page should contain a trailing space, unless it ends in a broken word.")
    tag: str | None = Field(..., title="Paragraph Tag",
                            description="An optional tag indicating the type of paragraph per custom user-provided instructions")


class ChapterContent(BaseModel):
    title: str = Field(..., title="Chapter Title", description="The chapter title")
    paragraphs: List[ImageOCRParagraph] = Field(..., title="Paragraphs",
                                                description="All paragraphs in the chapter")

class ImageOCRPage(BaseModel):
    page_number: int | None = Field(..., title="Page Number",
                                    description="The page number, if visible or can be deduced")
    chapter_title: str | None = Field(..., title="Chapter Title",
                                      description="The chapter title, if visible or can be deduced")
    paragraphs: List[ImageOCRParagraph] = Field(..., title="Paragraphs",
                                                description="List of paragraphs extracted from the page")


class ImageOCRResult(BaseModel):
    pages: List[ImageOCRPage] = Field(..., title="Pages",
                                      description="List of pages on the image (usually one or two)")
