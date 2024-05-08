import argparse
import io
from itertools import islice
import os
import time
import fitz

from dotenv import load_dotenv
from openai import OpenAI

from .utils import io2uri

from typing import Optional


load_dotenv()


client = OpenAI()


def run_analysis(
    oai_client: OpenAI,
    prompt: Optional[str] = None,
    img_uri: Optional[str] = None,
    metadata: dict = {},
):
    prompt = prompt.format(**metadata)
    # print("Request analysis with prompt: \"%s\"" % prompt)

    # chat_history = []
    # chat_history.append({})

    response = oai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_uri,
                        },
                    },
                ],
            },
        ],
        max_tokens=2400,
    )

    print("Output:")
    print(response.choices[0].message.content)

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--output-image")
    parser.add_argument("-d", "--dpi", type=int, default=300)
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-o", "--output-response")
    parser.add_argument("document")
    args = parser.parse_args()

    pdffile: str = args.document
    output_image_format: Optional[str] = args.output_image
    output_file_format: Optional[str] = args.output_response
    prompt_file: Optional[str] = args.prompt_file
    image_dpi: int = args.dpi

    prompt: Optional[str] = None
    if prompt_file is not None:
        with open(prompt_file, "r") as pf:
            prompt = pf.read()

    pdf_basename: str = os.path.basename(pdffile)
    previous_info: Optional[str] = None
    start_time_epoch = int(time.time())

    with fitz.open(pdffile) as doc:
        for page in islice(doc, 2):
            pix = page.get_pixmap(dpi=image_dpi)

            if output_image_format is not None:
                save_path = output_image_format.format(
                    page_number=page.number,
                    source_basename=pdf_basename,
                    start_time_epoch=start_time_epoch,
                )
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                pix.save(save_path)

            # Image inference
            print("Starting inference...")

            with io.BytesIO() as f:
                # Write image to buffer
                pix.pil_save(f, format="png")
                # Convert bufferred image to data uri
                img_data_uri = io2uri(f)
                metadata = {
                    "filename": pdf_basename,
                    "page_number": (page.number + 1),
                    "previous_info": previous_info or "(Nothing found yet.)",
                }

                response = run_analysis(
                    client,
                    prompt,
                    img_data_uri,
                    metadata=metadata,
                )

                if response is not None:
                    if previous_info is None:
                        previous_info = ""

                    previous_info += "{header}\n{content}\n------\n".format(
                        header="Page {page_number}:".format(**metadata),
                        content=response.strip(),
                    )

        print()
        print("Final output:")
        print(previous_info)

        if output_file_format is not None:
            if previous_info is not None:
                save_path = output_file_format.format(
                    source_basename=pdf_basename, start_time_epoch=start_time_epoch
                )
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(previous_info)
            else:
                print("Warning: No output was generated. Skipping writing to file.")
