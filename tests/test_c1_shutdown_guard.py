import unittest
import json
import tempfile
from pathlib import Path
import torch
from watch_c1_training_shutdown import other_training,validate_saved_run,RUN,MANIFEST_SHA


class TrainingGuardTests(unittest.TestCase):
    def test_other_training_blocks_shutdown(self):
        rows=[dict(ProcessId=1,CommandLine='python train_c1_yolo26s_joint_pretrained.py'),
              dict(ProcessId=2,CommandLine='python resume_c1_yolo26s_full60.py'),
              dict(ProcessId=3,CommandLine='python watch_c1_training_shutdown.py'),
              dict(ProcessId=4,CommandLine=None)]
        self.assertEqual([r['ProcessId'] for r in other_training(rows)],[1,2])

    def test_only_final_readable_checkpoints_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'weights').mkdir()
            (run/'data_receipt.json').write_text(json.dumps(dict(experiment=RUN.name,manifest_sha256=MANIFEST_SHA)))
            (run/'results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.3\n')
            model=torch.nn.Linear(2,2);model.names={0:'enemy',1:'teammate'}
            for name in ('best.pt','last.pt'):
                torch.save(dict(epoch=-1,optimizer=None,model=model),run/'weights'/name)
            self.assertEqual(validate_saved_run(run)['completed_epochs'],1)
            torch.save(dict(epoch=0,optimizer={},model=model),run/'weights/last.pt')
            with self.assertRaises(ValueError):validate_saved_run(run)
            (run/'weights/last.pt').write_bytes(b'incomplete checkpoint')
            with self.assertRaises(Exception):validate_saved_run(run)


if __name__=='__main__':unittest.main()
