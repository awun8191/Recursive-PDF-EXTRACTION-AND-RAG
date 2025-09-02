import json
import os
import os
from typing import Dict, Any, Optional, List

import chromadb
from chromadb import Settings
from pydantic import BaseModel

import src.data_models.question_model
import src.services.Gemini.gemini_service as gemini_service
import src.data_models.gemini_config as config
import src.data_models.course_model as course_model
from src.data_models import question_model


class ParseCourses:

    def __init__(self, path_to_course: str):
        self.course_path = path_to_course
        self.parsed_data = self._parse()

    def _parse(self):
        with open(self.course_path) as courses:
            data = json.load(courses)
        return data



    def get_course_info(self) -> list[course_model.CourseModel]:
        data1 = []
        for course in self.parsed_data:
            sem = course.get("semesters", "").strip()
            if sem == "FIRST":
                semester = "1"
            else:
                semester = "2"
            data1.append(
                course_model.CourseModel(
                    code=course["code"],
                    units=course["units"],
                    type=course["type"],
                    title=course["title"],
                    department=course["offered_by_programs"],
                    semester = semester,
                )
            )
        return data1


class QuestionRAG:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        gemini_config = config.GeminiConfig(temperature=0.5, response_schema=question_model.QuestionSet)
        self.gemini_service = gemini_service.GeminiService(model=model, generation_config=gemini_config)

        self.courses = ParseCourses(
            r"C:\Users\awun8\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\textbooks\courses.json").get_course_info()



    def _get_chroma_client(self, persist_dir: Optional[str] = None):
        """
        Create a Chroma client pointing to the persisted DB directory.
        Tries PersistentClient first, then falls back to Client(Settings(...)).
        """
        base_dir = self._resolve_persist_dir(persist_dir)
        try:
            # Newer Chroma API
            return chromadb.PersistentClient(path=base_dir)
        except Exception:
            # Fallback to older API style
            return chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=base_dir,
                    anonymized_telemetry=False,
                )
            )

    def _resolve_persist_dir(self, persist_dir: Optional[str]) -> str:
        """
        Resolve the Chroma persist directory. If not provided, defaults to <project-root>/chroma_db_bge_m3.
        """
        if persist_dir:
            return persist_dir
        here = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(here, "../../.."))
        return os.path.join(project_root, "chroma_db_bge_m3")

    def _resolve_collection(self, client, collection_name: Optional[str]):
        """
        Resolve a collection by name or default to the first available collection.
        """
        if collection_name:
            return client.get_collection(collection_name)
        collections = client.list_collections()
        if not collections:
            raise ValueError("No Chroma collections found in the persist directory.")
        first = collections[0]
        try:
            # list_collections may return objects with .name
            return client.get_collection(first.name)
        except Exception:
            # Or it may already be a Collection object
            return first

    def _retrieve_embeddings(
        self,
        metadata: Dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
        include_embeddings: bool = False,
        persist_dir: Optional[str] = None,
    ):
        """
        Backwards-compatible helper that delegates to find_embeddings_by_metadata.
        """
        include: List[str] = ["metadatas", "documents"]
        if include_embeddings:
            include.append("embeddings")
        return self.find_embeddings_by_metadata(
            metadata=metadata,
            collection_name=collection_name,
            limit=limit,
            include=include,
            persist_dir=persist_dir,
        )

    def find_embeddings_by_metadata(
        self,
        metadata: Dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
        include: Optional[List[str]] = None,
        persist_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find items in the Chroma DB by metadata filter.

        Args:
            metadata: Key/value metadata filter, e.g., {"course_code": "CHE101"}.
            collection_name: Optional specific collection name. If omitted, searches across all collections.
            limit: Max number of items to return.
            include: Which fields to include from Chroma ("metadatas", "documents", "embeddings").
            persist_dir: Path to the persisted Chroma directory (defaults to <project-root>/chroma_db_bge_m3).

        Returns:
            A list of dicts with id, and any included fields.
        """
        client = self._get_chroma_client(persist_dir)
        include_fields = include or ["metadatas", "documents"]

        # Resolve target collections
        target_collections: List = []
        if collection_name:
            target_collections = [client.get_collection(collection_name)]
        else:
            target_collections = client.list_collections()
            if not target_collections:
                resolved_dir = self._resolve_persist_dir(persist_dir)
                raise ValueError(
                    f"No Chroma collections found in persist directory: {resolved_dir}. "
                    f"Ensure the path is correct and the DB contains at least one collection."
                )

        # Aggregate results from one or more collections
        items: List[Dict[str, Any]] = []
        for col in target_collections:
            try:
                results = col.get(where=metadata, include=include_fields)
            except Exception:
                # If some collection type differs, skip gracefully
                continue

            ids = results.get("ids", []) or []
            metadatas = results.get("metadatas", []) or []
            documents = results.get("documents", []) or []
            embeddings = results.get("embeddings", []) or []

            for i, id_ in enumerate(ids):
                item: Dict[str, Any] = {"id": id_}
                if metadatas:
                    item["metadata"] = metadatas[i]
                if documents:
                    item["document"] = documents[i]
                if embeddings:
                    item["embedding"] = embeddings[i]
                items.append(item)

            if len(items) >= limit:
                break

        return items[:limit]



    def generate_questions(self, context: str):
        print(f"Topic: {self.courses[-1].title}")
        prompt = (
            f"{context}"
            f"The text provided above is the necessary context for generating questions. on the said topic"
            f"Generate 6 multiple-choice questions for the course '{self.courses[-1].title}'."
            f"Do not make reference to the context above. "
            "Return ONLY valid JSON that matches this schema: "
            "{ 'questions': [ { "
            "'course_code': string, 'course_name': string, 'topic_name': string, "
            "'difficulty_ranking': integer (1-10), 'difficulty': string, "
            "'question': string, 'options': [string], 'correct_answer': string, "
            "'explanation': string, 'solution_steps': [string] } ] }. "
            "Ensure options are A–D and correct_answer is one of 'A','B','C','D'."
        )
        output = self.gemini_service.generate(
            prompt=prompt,
            response_model=question_model.QuestionSet
        )
        print(output)



# data = QuestionRAG().find_embeddings_by_metadata(metadata={"COURSE_FOLDER": "EEE 313"})
# print(data)
QuestionRAG().generate_questions(context= r'''
 
Derivation 
The Newton-Raphson method is based on the principle that if the initial guess of the root of 
0
)
(

x
f
 is at 
ix , then if one draws the tangent to the curve at 
)
(
ix
f
, the point 
1

ix
 where 
the tangent crosses the x -axis is an improved estimate of the root (Figure 1). 
Using the definition of the slope of a function, at 
ix
x 
 

θ
 = 
x
f
i
tan

 

1
0



i
i
i
x
x
x
f
 = 
, 
which gives 


i
i
i
i
x
f
x
f
 = x
x


1
 
 
 
 
 
 
 
(1) 

03.04.2 
                                                       Chapter 03.04 
Equation (1) is called the Newton-Raphson formula for solving nonlinear equations of the 
form 

0

x
f
.  So starting with an initial guess, 
ix , one can find the next guess, 
1

ix
, by 
using Equation (1).  One can repeat this process until one finds the root within a desirable 
tolerance. 
 
Algorithm 
The steps of the Newton-Raphson method to find the root of an equation 
0

x
f
  are 
1. Evaluate 

x
f 
 symbolically 
2. Use an initial guess of the root, 
ix , to estimate the new value of the root, 
1

ix
, as 
             


i
i
i
i
x
f
x
f
 = x
x


1
 
3. Find the absolute relative approximate error 
a
 as 
            
0
10
1
1





i
i
i
a
x
 x
x
 = 
 
4. Compare the absolute relative approximate error with the pre-specified relative 
error tolerance, 
s
.  If 
a
>
s
, then go to Step 2, else stop the algorithm.  Also, 
check if the number of iterations has exceeded the maximum number of iterations 
allowed.  If so, one needs to terminate the algorithm and notify the user. 
 
                       
 
                           Figure 1  Geometrical illustration of the Newton-Raphson method. 
 
f (x) 
f (xi) 
f (xi+1) 
    xi+2     xi+1 
    xi 
    x 
    θ 
[xi,  f (xi)] 

Newton-Raphson Method                                                                                               03.04.3 
Example 1 
You are working for ‘DOWN THE TOILET COMPANY’ that makes floats for ABC 
commodes.  The floating ball has a specific gravity of 0.6 and has a radius of 5.5 cm.  You 
are asked to find the depth to which the ball is submerged when floating in water. 
 
                                      
 
                                          Figure 2   Floating ball problem. 
 
The equation that gives the depth x  in meters to which the ball is submerged under water is 
given by 
0
10
993
.3
165
.0
4
2
3





x
x
 
Use the Newton-Raphson method of finding roots of equations to find  
a) the depth x  to which the ball is submerged under water.  Conduct three iterations 
to estimate the root of the above equation.   
b) the absolute relative approximate error at the end of each iteration, and  
c) the number of significant digits at least correct at the end of each iteration. 
Solution 

4
2
3
10
993
.3
165
0





x
.
x
x
f
 

x
.
x
x
f
33
0
3
2 


 
Let us assume the initial guess of the root of 

0

x
f
 is 
.
.
x
m
 
05
0
0 
  This is a reasonable 
guess (discuss why 
0

x
 and 
m
 
11
.0

x
 are not good choices) as the extreme values of the 
depth x  would be 0 and the diameter (0.11 m) of the ball.   
Iteration 1  
The estimate of the root is 




0
0
0
1
x
f
x
f
x
x



 
    








05
0
33
0
05
0
3
10
993
.3
05
0
165
0
05
0
05
0
2
4
2
3
.
.
.
.
.
.
.







 
    
3
4
10
9
10
118
.1
05
0






.
 
    


01242
.0
05
0


.
 
                
06242
0.

  

03.04.4 
                                                       Chapter 03.04 
The absolute relative approximate error 
a
 at the end of Iteration 1 is 
100
1
0
1




x
x
x
a
 
      
19.90%
 
100
06242
0
05
0
06242
0




.
.
.
 
        
The number of significant digits at least correct is 0, as you need an absolute relative 
approximate error of 5% or less for at least one significant digit to be correct in your result. 
Iteration 2 
The estimate of the root is 


1
1
1
2
x
f
x
f
x
x



 
     








06242
0
33
0
06242
0
3
10
993
.3
06242
0
165
0
06242
0
06242
0
2
4
2
3
.
.
.
.
.
.
.







 
     
3
7
10
90973
.8
10
97781
3
06242
0








.
.
 
     


5
10
4646
.4
06242
0



.
 
     
06238
0.

 
''')




