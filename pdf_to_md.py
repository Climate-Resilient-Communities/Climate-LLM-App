### First File
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Iterable
from tqdm.auto import tqdm

from docling.datamodel.base_models import ConversionStatus, PipelineOptions
from docling.datamodel.document import ConversionResult, DocumentConversionInput
from docling.document_converter import DocumentConverter

def export_documents(conv_results: Iterable[ConversionResult], output_dir: Path) -> list, list, list:
    """
    export_documents returns a list of successfully, partially converted, and failed documents after PDF to MD conversation using docling
    It takes the docling conversion results and writes successfully converted markdown files into designated directory
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists
    success_lst = []
    failure_lst = []
    partial_success_lst = []

    # Categorize conversion results 
    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            doc_filename = conv_res.input.file.stem
            success_lst.append(doc_filename)

            # Export Markdown format:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(conv_res.render_as_markdown())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_lst.append(conv_res.input.file)

        else:
            failure_lst.append(conv_res.input.file)


    print(
        f"Processed {len(success_lst) + len(partial_success_lst) + len(failure_lst)} docs, "
        f"of which {failure_lst} failed "
        f"and {partial_success_lst} were partially converted."
    )
    return success_lst, partial_success_lst, failure_lst

def batch_convert_documents(PDF_dir: Path, MD_dir: Path, batch_size: int):

    """
    batch_convert_documents returns a list of successfully, partially converted, and failed documents after PDF to MD conversation using docling.
    It takes PDFs' directory path PDF_dir and and writes successfully converted markdown files into designated directory MD_dir in batches based on batch_size
    """
    # Find all the PDF files in the PDF_dir
    PDFs_file_path = []
    for root, dirs, files in os.walk(PDF_dir):
        PDF_files = [Path(os.path.join(root, f)) for f in files if f.endswith('.pdf')]
        PDFs_file_path.extend(PDF_files)
    
    # Initialize lists
    all_success_lst = []
    all_failure_lst = []
    all_partial_success_lst = []

    # Initialize docling DocumentConverter
    doc_converter = DocumentConverter()

    # Time the PDF to MD conversion process 
    start_time = time.time()

    # Convert PDFs into MDs in batches 
    for i in tqdm(range(0, len(PDFs_file_path), batch_size)):
        i_end = min(i + batch_size, len(PDFs_file_path))
        batch_PDFs_file_path = PDFs_file_path[i:i_end]
    

        
        input = DocumentConversionInput.from_paths(batch_PDFs_file_path)
        conv_results = doc_converter.convert(input)
        success_lst, partial_success_lst, failure_lst = export_documents(
            conv_results, output_dir=Path(MD_dir)
        )

        all_success_lst.extend(success_lst)
        all_failure_lst.extend(failure_lst)
        all_partial_success_lst.extend(partial_success_lst)


    end_time = time.time() - start_time
    print((f"All documents were converted in {end_time:.2f} seconds."))


    return success_lst, partial_success_lst, failure_lst

if __name__ == "__main__":
    PDF_dir = sys.argv[1]
    MD_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    success_lst, partial_success_lst, failure_lst = batch_convert_documents(PDF_dir, MD_dir, batch_size)
    print(success_lst, partial_success_lst, failure_lst)
