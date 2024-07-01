# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from absl.testing import absltest

import google
import google.generativeai as genai
import pathlib

media = pathlib.Path(__file__).parents[1] / "third_party"


class UnitTests(absltest.TestCase):
    def test_files_create(self):
        # [START files_create_text]
        myfile = genai.upload_file(media / "poem.txt")
        print(f"{myfile=}")

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(
            [myfile, "\n\n", "Can you add a few more lines to this poem?"]
        )
        print(f"{result.text=}")
        # [END files_create_text]

    def test_files_create_audio(self):
        # [START files_create_audio]
        myfile = genai.upload_file(media / "sample.mp3")
        print(f"{myfile=}")

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([myfile, "Describe this audio clip"])
        print(f"{result.text=}")
        # [END files_create_audio]

    def test_files_create_video(self):
        # [START files_create_video]
        import time

        # Video clip (CC BY 3.0) from https://peach.blender.org/download/
        myfile = genai.upload_file(media / "Big_Buck_Bunny.mp4")
        print(f"{myfile=}")

        # Videos need to be processed before you can use them.
        while myfile.state.name == "PROCESSING":
            print("processing video...")
            time.sleep(5)
            myfile = genai.get_file(myfile.name)

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([myfile, "Describe this video clip"])
        print(f"{result.text=}")
        # [END files_create_video]

    def test_files_list(self):
        # [START files_list]
        print("My files:")
        for f in genai.list_files():
            print("  ", f.name)
        # [END files_list]

    def test_files_delete(self):
        # [START files_delete]
        myfile = genai.upload_file(media / "poem.txt")

        myfile.delete()

        try:
            # Error.
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([myfile, "Describe this file."])
        except google.api_core.exceptions.PermissionDenied:
            pass
        # [END files_delete]

    def test_files_get(self):
        # [START files_get]
        myfile = genai.upload_file(media / "poem.txt")
        file_name = myfile.name
        print(file_name)  # "files/*"

        myfile = genai.get_file(file_name)
        print(myfile)
        # [END files_get]


if __name__ == "__main__":
    absltest.main()
